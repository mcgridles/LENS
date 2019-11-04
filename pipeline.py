import sys
import os
import argparse
import cv2
import numpy as np
import torch.multiprocessing as mp
import pickle

from utils import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'two-stream-action-recognition'))
sys.path.append(os.path.join(ROOT_DIR, 'flownet2-pytorch'))
from action_recognition import SpatialCNN, MotionCNN
from optical_flow import OpticalFlow, tools

mp.set_start_method('spawn', force=True)  # Set multiprocessing start method for CUDA


def inference(optical_flow, spatial_cnn, motion_cnn, args):
    """
    Perform inference on a video or stream.

    :param optical_flow: (OpticalFlow) -> FlowNet2.0 wrapper object for performing optical flow inference
    :param spatial_cnn: (SpatialCNN) -> Spatial CNN wrapper object for performing spatial inference
    :param motion_cnn: (MotionCNN) -> Motion CNN wrapper object for performing temporal inference
    :param args: (argparse.args) -> Command line arguments
    :return: (list(list(float))) -> List of class predictions for each frame
    """

    print('Starting inference')

    # Initialize queues
    frame_queue = mp.Queue(maxsize=10)
    flow_queue = mp.Queue(maxsize=10)
    spatial_pred_queue = mp.Queue()
    motion_pred_queue = mp.Queue()

    # Initialize and start action recognition processes
    spatial_process = mp.Process(target=spatial_cnn.run_async, args=(frame_queue, spatial_pred_queue))
    motion_process = mp.Process(target=motion_cnn.run_async, args=(flow_queue, motion_pred_queue))
    spatial_process.start()
    motion_process.start()

    cap = cv2.VideoCapture(args.stream)

    # Calculate render size for initializing optical flow array
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    render_size = args.inference_size
    if (render_size[0] < 0) or (render_size[1] < 0) or (frame_size[0] % 64) or (frame_size[1] % 64):
        render_size[0] = ((frame_size[0]) // 64) * 64
        render_size[1] = ((frame_size[1]) // 64) * 64

    of = np.tile(np.zeros(render_size), (20, 1, 1))
    rgb = np.zeros((11, frame_size[0], frame_size[1], 3)).astype(np.uint8)
    predictions = []

    prev_frame = None
    frame_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with tools.TimerBlock('Processing frame {}'.format(frame_counter)) as block:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb[-1, :, :, :] = frame

                # Run optical flow starting at second frame
                if prev_frame is not None and frame is not None:
                    flow = optical_flow.run([prev_frame, frame])
                    of[-2:, :, :] = flow
                    block.log('Optical flow complete')

                # Start making predictions at 11th frame
                if frame_counter >= 10:
                    # Put current frame and optical flow on respective queues
                    frame_queue.put(rgb[0, :, :, :])
                    flow_queue.put(of)

                    # Wait for predictions
                    spatial_preds = spatial_pred_queue.get(block=True)
                    block.log('Spatial predictions complete')
                    motion_preds = motion_pred_queue.get(block=True)
                    block.log('Motion predictions complete')

                    # Add predictions
                    predictions.append(spatial_preds + motion_preds)

                # Put flow at end of array and rotate to make room for the next one
                # Once array is full the first one will cycle back to end and be overwritten
                rgb = np.roll(rgb, -1)
                of = np.roll(of, -2)

                prev_frame = frame
                frame_counter += 1
    finally:
        # Catch any exceptions to prevent processes from hanging
        # Break out of loops and join processes
        frame_queue.put(-1)
        flow_queue.put(-1)
        spatial_process.join()
        motion_process.join()

    return predictions


def parse_args():
    """
    Parse command line arguments.

    :return: (argparse.args) -> Argument object
    """

    parser = argparse.ArgumentParser('LENS Pipeline', parents=[flow_parser(), spatial_parser(), motion_parser()])

    # Video stream
    parser.add_argument('--stream', '-s', type=str, help='Path to video stream', default='')
    parser.add_argument('--nb-classes', type=int, metavar='N', help='Number of action classes', default=4)

    args = parse_flow_args(parser)

    return args


def main():
    """
    Command for running on capstone4790-vm-1 (IP: 35.197.106.62):
    >>> python pipeline.py --stream /mnt/disks/datastorage/videos/keyboard_cat.mp4 \
                           -ow /mnt/disks/datastorage/weights/optical_weights.pth.tar \
                           -sw /mnt/disks/datastorage/weights/spatial_weights.pth.tar \
                           -mw /mnt/disks/datastorage/weights/motion_weights.pth.tar
    """
    args = parse_args()

    optical_flow = OpticalFlow(args)
    spatial_cnn = SpatialCNN(args)
    motion_cnn = MotionCNN(args)

    predictions = inference(optical_flow, spatial_cnn, motion_cnn, args)

    video_name = os.path.splitext(os.path.basename(args.stream))[0]
    video_dir = os.path.dirname(os.path.abspath(args.stream))
    pickle_file = os.path.join(video_dir, '{}_predictions.pkl'.format(video_name))

    # Save predictions in pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(predictions, f)

if __name__ == '__main__':
    main()
