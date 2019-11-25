import sys
import os
import argparse
import cv2
import numpy as np
import torch.multiprocessing as mp
import pickle
import sklearn
import time
from datetime import datetime
from scipy.special import logsumexp

from utils import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'two-stream-action-recognition'))
sys.path.append(os.path.join(ROOT_DIR, 'flownet2-pytorch'))
sys.path.append(os.path.join(ROOT_DIR, 'capstone-lens'))

from action_recognition import SpatialCNN, MotionCNN
from optical_flow import OpticalFlow, tools
from ExternalMessages import SendUtility

mp.set_start_method('spawn', force=True)  # Set multiprocessing start method for CUDA


def inference(optical_flow, spatial_cnn, motion_cnn, svm_model, args):
    """
    Perform inference on a video or stream.

    :param optical_flow: (OpticalFlow) -> FlowNet2.0 wrapper object for performing optical flow inference
    :param spatial_cnn: (SpatialCNN) -> Spatial CNN wrapper object for performing spatial inference
    :param motion_cnn: (MotionCNN) -> Motion CNN wrapper object for performing temporal inference
    :param args: (argparse.args) -> Command line arguments
    :return: (list(list(float))) -> List of class predictions for each frame
    """

    print('Starting inference')

    cap = cv2.VideoCapture(args.stream)
    if not cap.isOpened():
        print('Could not open video stream: {}'.format(args.stream))
        exit()
        
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate render size for initializing optical flow array
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    output_dir = '/tmp'
    sender = SendUtility(output_dir, save_buffer)
    sender.start()

    buffer_size = 10
    buf = np.zeros((buffer_size, frame_size[0], frame_size[1], 3), dtype=np.uint8)
    
    predictions = []
    prev_frame = None
    frame_counter = 0
    try:
        t_start = time.time()
        while True:
            ret = cap.grab()
            if frame_counter % (args.skip_frames + 1) == 0:
                ret, frame = cap.retrieve()
            else:
                frame_counter += 1
                continue
                
            if not ret:
                break

            with tools.TimerBlock('Processing frame {}'.format(frame_counter), False) as block:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                buf[-1, :, :, :] = frame
                
                spatial_preds = spatial_cnn.run(frame)

                # Run optical flow starting at second frame
                if prev_frame is not None and frame is not None:
                    flow = optical_flow.run([prev_frame, frame])
                    motion_preds = motion_cnn.run(flow)

                if spatial_preds is not None and motion_preds is not None:
                    preds = combine_predictions(spatial_preds, motion_preds, svm_model)
                    predictions.append(preds)

                    sender.add_to_queue(buf, preds.squeeze(0))

                buf = np.roll(buf, -1, axis=0)
                prev_frame = frame
                frame_counter += 1
    finally:
        cap.release()
        
        t_end = time.time() - t_start
        print('\nFPS: {0}'.format(round(num_frames / t_end, 2)))
              
    return predictions


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def combine_predictions(spatial_preds, motion_preds, svm_model=None):
    spatial_softmax = softmax(spatial_preds, axis=1)
    motion_softmax = softmax(motion_preds, axis=1)
        
    if svm_model is not None:
        predictions = np.hstack((spatial_softmax, motion_softmax))
        return svm_model.predict_proba(predictions)
    else:
        return (spatial_softmax + motion_softmax) / 2


def parse_args():
    """
    Parse command line arguments.

    :return: (argparse.args) -> Argument object
    """

    parser = argparse.ArgumentParser('LENS Pipeline', parents=[flow_parser(), spatial_parser(), motion_parser()])

    # Video stream
    parser.add_argument('--stream', type=str, help='Path to video stream', default='')
    parser.add_argument('--svm', type=str, help='Path to saved SVM model', default='')
    parser.add_argument('--nb_classes', type=int, metavar='N', help='Number of action classes', default=4)
    parser.add_argument('--skip_frames', type=int, help='Number of frames to skip', default=1)
    parser.add_argument('--save', type=str, help='Path to directory to save output pickle file', default='')

    args = parse_flow_args(parser)
    
    try:
        # Camera streams are numbered, must be int
        args.stream = int(args.stream)
    except ValueError:
        pass

    return args


def main():
    """
    Command for running on capstone4790-vm-1 (IP: 35.197.106.62):
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

    optical_flow = OpticalFlow(args)
    spatial_cnn = SpatialCNN(args)
    motion_cnn = MotionCNN(args)
    
    if args.svm:
        print('Loading SVM')
        with open(args.svm, 'rb') as file:
            svm_model = pickle.load(file)
    else:
        svm_model = None

    predictions = inference(optical_flow, spatial_cnn, motion_cnn, svm_model, args)

    if args.save:
        timestamp = datetime.now().strftime('%m%d%y_%H%M%S')
        pickle_file = os.path.join(args.save, 'predictions_{}.pkl'.format(timestamp))

        # Save predictions in pickle file
        print('Saving predictions in {}'.format(pickle_file))
        with open(pickle_file, 'wb') as f:
            pickle.dump(predictions, f)

            
if __name__ == '__main__':
    main()
