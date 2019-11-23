import sys
import os
import argparse
import cv2
import numpy as np
import torch.multiprocessing as mp
import pickle
import time

from utils import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'two-stream-action-recognition'))
sys.path.append(os.path.join(ROOT_DIR, 'flownet2-pytorch'))
sys.path.append(os.path.join(ROOT_DIR, 'capstone-lens'))

from action_recognition import SpatialCNN, MotionCNN
from optical_flow import OpticalFlow, tools
from ExternalMessages import SendUtility

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

#     # Initialize queues
#     frame_queue = mp.Queue(maxsize=10)
#     flow_queue = mp.Queue(maxsize=10)
#     spatial_pred_queue = mp.Queue()
#     motion_pred_queue = mp.Queue()

#     # Initialize and start action recognition processes
#     spatial_process = mp.Process(target=spatial_cnn.run_async, args=(frame_queue, spatial_pred_queue))
#     motion_process = mp.Process(target=motion_cnn.run_async, args=(flow_queue, motion_pred_queue))
#     spatial_process.start()
#     motion_process.start()

    cap = cv2.VideoCapture(args.stream)

    # Calculate render size for initializing optical flow array
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    render_size = args.inference_size
    if (render_size[0] < 0) or (render_size[1] < 0) or (frame_size[0] % 64) or (frame_size[1] % 64):
        render_size[0] = ((frame_size[0]) // 64) * 64
        render_size[1] = ((frame_size[1]) // 64) * 64

    output_dir = '/tmp'
    sender = SendUtility(output_dir, save_buffer)
    sender.start()

    buffer_size = 10
    buf = np.zeros((buffer_size, frame_size[0], frame_size[1], 3), dtype=np.uint8)
    
    predictions = []
    prev_frame = None
    frame_counter = 0
    try:
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
                
#                 frame_queue.put(frame)
#                 spatial_preds = spatial_pred_queue.get(block=True)
                spatial_preds = spatial_cnn.run(frame)

                # Run optical flow starting at second frame
                if prev_frame is not None and frame is not None:
                    flow = optical_flow.run([prev_frame, frame])
                    
#                     flow_queue.put(flow)
#                     motion_preds = motion_pred_queue.get(block=True)
                    motion_preds = motion_cnn.run(flow)
                    

                if spatial_preds is not None and motion_preds is not None:
                    # Add predictions
                    preds = svm(spatial_preds, motion_preds)
                    predictions.append(preds)

                    sender.add_to_queue(buf, preds.squeeze(0))

                buf = np.roll(buf, -1, axis=0)
                prev_frame = frame
                frame_counter += 1
    finally:
        # Catch any exceptions to prevent processes from hanging
        # Break out of loops and join processes
#         frame_queue.put(-1)
#         flow_queue.put(-1)
#         spatial_process.join()
#         motion_process.join()
        cap.release()

    print(predictions)
    return predictions


def svm(spatial_preds, motion_preds):
    return spatial_preds + motion_preds


def frame_inference(optical_flow, args):
    image1 = cv2.imread(args.images[0])
    image2 = cv2.imread(args.images[1])
    
    height, width, _ = image1.shape
    
    # Calculate render size for initializing optical flow array
    frame_size = (height, width)
    render_size = args.inference_size
    if (render_size[0] < 0) or (render_size[1] < 0) or (frame_size[0] % 64) or (frame_size[1] % 64):
        render_size[0] = ((frame_size[0]) // 64) * 64
        render_size[1] = ((frame_size[1]) // 64) * 64

    
    flow = optical_flow.run([image1, image2])
    flow = flow.transpose(1, 2, 0)
    optical_flow.display_flow(flow, save_path='.')


def parse_args():
    """
    Parse command line arguments.

    :return: (argparse.args) -> Argument object
    """

    parser = argparse.ArgumentParser('LENS Pipeline', parents=[flow_parser(), spatial_parser(), motion_parser()])

    # Video stream
    parser.add_argument('--stream', type=str, help='Path to video stream', default='')
    parser.add_argument('--nb_classes', type=int, metavar='N', help='Number of action classes', default=4)
    parser.add_argument('--skip_frames', type=int, help='Number of frames to skip', default=1)
    parser.add_argument('--images', type=str, help='Path to test images', nargs=2, default=[])

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

    if len(args.images) > 0:
        frame_inference(optical_flow, args)
    else:
        predictions = inference(optical_flow, spatial_cnn, motion_cnn, args)

        video_name = os.path.splitext(os.path.basename(args.stream))[0]
        video_dir = os.path.dirname(os.path.abspath(args.stream))
        pickle_file = os.path.join(video_dir, '{}_predictions.pkl'.format(video_name))

        # Save predictions in pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump(predictions, f)

if __name__ == '__main__':
    main()
