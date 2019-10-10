import sys
import os
import argparse
import queue
import colorama
import cv2
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'two-stream-action-recognition'))
sys.path.append(os.path.join(ROOT_DIR, 'flownet2-pytorch'))

from action_recognition import SpatialCNN, MotionCNN
from optical_flow import OpticalFlow
from optical_flow import models, losses, tools


def inference(cap, optical_flow, spatial_cnn, motion_cnn):
    frame_queue = queue.queue(maxsize=11)
    predictions = []

    ret = True
    while ret:
        ret, frame = cap.read()
        frame_queue.put(frame)

        if frame_queue.full():
            of = []
            for i in range(frame_queue.qsize()/2):
                frame1 = frame_queue.get()
                frame2 = frame_queue.get()

                # Put frames back on queue so it's a moving window
                frame_queue.put(frame1)
                frame_queue.put(frame2)

                of.append(optical_flow.run([frame1, frame2]))

            spatial_preds = spatial_cnn.run(frame)
            temporal_preds = motion_cnn.run(of)  # I'm not quite sure whether this should be all 10 flows or just one
            predictions.append(spatial_preds + temporal_preds)
            
            frame_queue.get()  # Remove first frame to make room for next one

    return predictions


def parse_args():
    """
    Parse and prepare command line arguments.
    """

    parser = argparse.ArgumentParser()
    flow = parser.add_argument_group('optical flow')
    spatial = parser.add_argument_group('spatial')
    motion = parser.add_argument_group('motion')

    # Video stream
    parser.add_argument('--stream', '-s', type=str, help='path to video stream', default='')

    ### FlowNet args ###
    # CUDA
    flow.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    flow.add_argument('--no_cuda', action='store_true')

    # Preprocessing
    flow.add_argument('--seed', type=int, default=1)
    flow.add_argument('--rgb_max', type=float, default=255.0)
    flow.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    flow.add_argument('--fp16_scale', type=float, default=1024.0,
        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    flow.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
        help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')

    # Weights
    flow.add_argument('--optical_weights', '-ow', type=str, help='path to FlowNet weights', default='')

    ### Spatial args ###
    spatial.add_argument('--spatial_weights', '-sw', type=str, help='path to spatial CNN weights', default='')

    ### Motion args ###
    motion.add_argument('--motion_weights', '-mw', type=str, help='path to motion CNN weights', default='')

    ### Model and loss ###
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')
    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

    with tools.TimerBlock('Parsing Arguments') as block:
        args, unknown = parser.parse_known_args()
        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()

        # Have to do it this way since there seem to be issues using `required=True` in `add_argument()`
        if not (args.stream or args.optical_weights or args.spatial_weights or args.motion_weights):
            raise Exception('Video stream and weights are required')

        # Print all arguments, color the non-defaults
        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.loss_class = tools.module_to_dict(losses)[args.loss]
        args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def main():
    """
    Command for running on capstone4790-vm-1 (IP: 35.197.106.62):
    >>> python pipeline.py --stream /mnt/disks/datastorage/videos/keyboard_cat.mp4 \
                           -ow /mnt/disks/datastorage/weights/flow_weights.pth.tar \
                           -sw /mnt/disks/datastorage/weights/spatial_weights.pth.tar \
                           -tw /mnt/disks/datastorage/weights/motion_weights.pth.tar
    """
    args = parse_args()

    cap = cv2.VideoCapture(args.stream)
    optical_flow = OpticalFlow(args)
    spatial_cnn = SpatialCNN(args.spatial_weights)
    motion_cnn = MotionCNN(args.motion_weights)

    predictions = inference(cap, optical_flow, spatial_cnn, motion_cnn)
    

if __name__ == '__main__':
    main()
