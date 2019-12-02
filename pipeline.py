import os
import sys
import time
import pickle
import argparse
from datetime import datetime

import cv2
import sklearn
import numpy as np
from scipy.special import logsumexp

from utils import *

# Manually add paths to system path due to submodule imports being tricky
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'two-stream-action-recognition'))
sys.path.append(os.path.join(ROOT_DIR, 'flownet2-pytorch'))
sys.path.append(os.path.join(ROOT_DIR, 'LENS_Network'))

from action_recognition import SpatialCNN, MotionCNN
from optical_flow import OpticalFlow, tools
from ExternalMessages import SendUtility

DEFAULT_SVM_MODEL = '/mnt/disks/datastorage/weights/svm.pkl'
DEFAULT_PRED_DIR = '/mnt/disks/datastorage/predictions'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'protocol_whitelist;file,udp,rtp'  # For video streaming


class LENS:
    """
    Optical flow + two stream action recognition inference class.
    """

    def __init__(self, args):
        self.args = args
        self.optical_flow = None
        self.spatial_cnn = None
        self.motion_cnn = None
        self.svm_model = None
        self.cap = None
        self.sender = None
        self.buf = None

        self.load()

    def load(self):
        """
        Load models and objects needed for inference.

        :return: (None)
        """

        with tools.TimerBlock('Loading inference models', True) as block:
            self.spatial_cnn = SpatialCNN(self.args)

            # Using spatial network only speeds up inference
            if not self.args.spatial_only:
                self.optical_flow = OpticalFlow(self.args)
                self.motion_cnn = MotionCNN(self.args)
            else:
                block.log('Skipping temporal network')

            if self.args.svm:
                block.log('Loading SVM')
                with open(self.args.svm, 'rb') as file:
                    self.svm_model = pickle.load(file)

            self.cap = cv2.VideoCapture(self.args.stream)
            if not self.cap.isOpened():
                block.log('ERROR: couldn\'t open video stream: {}'.format(self.args.stream))
                self.cap.release()
                exit()

            output_dir = '/tmp'
            block.log('Initializing message utility')
            self.sender = SendUtility(output_dir, save_buffer)
            self.sender.start()

            buffer_size = self.args.buffer_size
            frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.buf = np.zeros((buffer_size, frame_size[0], frame_size[1], 3), dtype=np.uint8)

    def inference(self):
        """
        Perform inference on a video or stream.

        :return: (list(np.ndarray)) -> List of class predictions for each frame
        """

        print('Starting inference')
        
        predictions = []
        prev_frame = None
        frame_counter = 0
        try:
            t_start = time.time()
            while True:
                # Grab encoded data from stream
                ret = self.cap.grab()

                if frame_counter % (self.args.skip_frames + 1) == 0:
                    # Decode frame if not skipped
                    ret, frame = self.cap.retrieve()
                else:
                    frame_counter += 1
                    continue

                if not ret:
                    break

                with tools.TimerBlock('Processing frame {}'.format(frame_counter), False) as block:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.buf[-1, :, :, :] = frame

                    # Perform inference
                    if self.args.spatial_only:
                        preds = self.spatial_cnn(frame)
                        if preds is not None:
                            preds = self.softmax(preds, axis=1)
                    else:
                        preds = self._inference(frame, prev_frame)

                    # Send results to message handler over TCP
                    if preds is not None:
                        block.log('Predictions: {}'.format(preds))
                        predictions.append(preds)
                        self.sender.add_to_queue(self.buf, preds.squeeze(0))

                    # Roll buffer along first axis to prepare for next frame
                    self.buf = np.roll(self.buf, -1, axis=0)
                    prev_frame = frame
                    frame_counter += 1

            # Print timing info
            t_end = time.time() - t_start
            num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print('\nFPS: {0}'.format(round(num_frames / t_end, 2)))

        finally:
            self.cap.release()
                  
        return predictions

    def _inference(self, frame, prev_frame):
        """
        Inference implementation.

        :param frame: (np.ndarray) -> Current video frame
        :param prev_frame: (np.ndarray) -> Previous video frame
        :return: (np.ndarray) -> Prediction probabilities for each class
        """

        # Start at second frame because of optical flow
        preds = None
        if prev_frame is not None and frame is not None:
            spatial_preds = self.spatial_cnn(frame)

            flow = self.optical_flow([prev_frame, frame])
            motion_preds = self.motion_cnn(flow)

            if spatial_preds is not None and motion_preds is not None:
                preds = self.combine_predictions(spatial_preds, motion_preds)
            
        return preds

    def combine_predictions(self, spatial_preds, motion_preds):
        """
        Combine predictions from two stream network.

        :param spatial_preds: (np.ndarray) -> Prediction probabilities from spatial stream
        :param motion_preds: (np.ndarray) -> Prediction probabilities from temporal stream
        :return: (np.ndarray) -> Combined prediction probabilities
        """

        spatial_softmax = self.softmax(spatial_preds, axis=1)
        motion_softmax = self.softmax(motion_preds, axis=1)
            
        if self.svm_model is not None:
            preds = np.hstack((spatial_softmax, motion_softmax))
            return self.svm_model.predict_proba(preds)
        else:
            return (spatial_softmax + motion_softmax) / 2

    @staticmethod
    def softmax(x, axis=None):
        """
        Softmax function.

        Note: Function copied from newer version of SciPy.

        :param x: (np.ndarray) -> Original prediction probabilities
        :param axis: (int) -> Axis over which to perform softmax
        :return: (np.ndarray) -> Prediction probabilities scaled from 0 to 1
        """

        return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def parse_args():
    """
    Parse command line arguments.

    :return: (argparse.args) -> Argument object
    """

    parser = argparse.ArgumentParser('LENS Pipeline', parents=[flow_parser(), spatial_parser(), motion_parser()])

    # Video stream
    parser.add_argument('--stream', type=str, help='Path to video stream', default='')
    parser.add_argument('--svm', type=str, help='Path to saved SVM model', default=DEFAULT_SVM_MODEL)
    parser.add_argument('--save', type=str, help='Path to directory to save output pickle file', default=DEFAULT_PRED_DIR)
    parser.add_argument('--nb_classes', type=int, metavar='N', help='Number of action classes', default=4)
    parser.add_argument('--skip_frames', type=int, help='Number of frames to skip', default=1)
    parser.add_argument('--spatial_only', action='store_true', help='Run using only the spatial network')
    parser.add_argument('--buffer_size', type=int, help='Length of saved clip buffer', default=10)

    args = parse_flow_args(parser)
    
    try:
        # Camera streams are numbered, must be int
        args.stream = int(args.stream)
    except ValueError:
        pass

    return args


def save_predictions(predictions, save_dir):
    """
    Save predictions in pickle file.

    :param predictions: (np.ndarray) -> Predictions for entire video
    :param save_dir: (str) -> Path to directory to save file in
    :return: (None)
    """

    timestamp = datetime.now().strftime('%m%d%y_%H%M%S')
    pickle_file = os.path.join(save_dir, 'predictions_{}.pkl'.format(timestamp))

    # Save predictions in pickle file
    print('Saving predictions in {}'.format(pickle_file))
    with open(pickle_file, 'wb') as f:
        pickle.dump(predictions, f)


def main():
    """
    >>> python pipeline.py --stream /path/to/video.mov \
                           --model FlowNet2CSS \
                           --svm /path/to/svm/model.pkl \
                           --nb_classes 4 \
                           --skip_frames 1 \
                           --save /path/to/directory/ \
                           --optical_weights /path/to/optical_weights.pth.tar \
                           --spatial_weights /path/to/spatial_weights.pth.tar \
                           --motion_weights /path/to/motion_weights.pth.tar
    """
    
    args = parse_args()
    lens = LENS(args)
    predictions = lens.inference()

    if args.save:
        save_predictions(predictions, args.save)

            
if __name__ == '__main__':
    main()
