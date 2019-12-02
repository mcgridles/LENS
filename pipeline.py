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

    def __del__(self):
        self.cap.release()

    def load(self):
        """
        Load models and objects needed for inference.

        :return: (None)
        """

        self.spatial_cnn = SpatialCNN(self.args)

        # Using spatial network only speeds up inference
        if not self.args.spatial_only:
            self.optical_flow = OpticalFlow(self.args)
            self.motion_cnn = MotionCNN(self.args)
        else:
            block.log('Skipping temporal network')

        if self.args.svm:
            with tools.TimerBlock('Building SVM model', True) as block:
                block.log('Loading weights {}'.format(self.args.svm))
                with open(self.args.svm, 'rb') as file:
                    self.svm_model = pickle.load(file)

        with tools.TimerBlock('Opening video stream', True) as block:
            self.cap = cv2.VideoCapture(self.args.stream)
            if self.cap.isOpened():
                block.log('Successfully connected to stream {}'.format(self.args.stream))
            else:
                block.log('Could not open video stream {}'.format(self.args.stream))
                self.cap.release()
                exit()

        with tools.TimerBlock('Setting up message handler', True) as block:
            output_dir = '/tmp'

            block.log('Initializing message utility')
            self.sender = SendUtility(output_dir, save_buffer)
            self.sender.start()

            block.log('Initializing frame buffer')
            buffer_size = self.args.buffer_size
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.buf = np.zeros((buffer_size, frame_height, frame_width, 3), dtype=np.uint8)

            block.log('Saving videos to {}'.format(output_dir))

    def inference(self):
        """
        Perform inference on a video or stream.

        :return: (list(np.ndarray)) -> List of class predictions for each frame
        """

        print('Starting inference')
        
        predictions = []
        prev_frame = None
        frame_counter = 0

        t_start = time.time()
        while True:
            ret, frame = self.get_frame(frame_counter)
            frame_counter += 1
            if not ret:
                break
            elif frame is None:
                continue

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

        # Print timing info
        t_end = time.time() - t_start
        num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('\n{0}{1}'.format('Framerate: '.ljust(15, ' '), round(num_frames / t_end, 2)))
                  
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

    def get_frame(self, frame_counter):
        """
        Get current frame from stream.

        :param frame_counter: (int) -> Current frame index
        :return: (bool)       -> Valid frame flag, False if no frame grabbed
                 (np.ndarray) -> Current frame
        """

        # Grab encoded data from stream
        ret = self.cap.grab()
        frame = None

        if frame_counter % (self.args.skip_frames + 1) == 0:
            # Decode frame if not skipped
            ret, frame = self.cap.retrieve()

        return ret, frame

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
    parser.add_argument('--svm', type=str, help='Path to saved SVM model', default='')
    parser.add_argument('--save', type=str, help='Path to directory to save output pickle file', default='')
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
    print('{0}{1}\n'.format('Predictions: '.ljust(15, ' '), pickle_file))
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
