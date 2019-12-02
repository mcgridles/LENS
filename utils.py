import os
import sys
import colorama
import torch
import argparse
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'two-stream-action-recognition'))
sys.path.append(os.path.join(ROOT_DIR, 'flownet2-pytorch'))
from optical_flow import models, losses, tools


def flow_parser():
    """
    Create argument parser for optical flow class (inference only).

    :return: (argparse.ArgumentParser) -> Parser with arguments for optical flow
    """

    parser = argparse.ArgumentParser(add_help=False)

    # CUDA
    parser.add_argument('--number_gpus', type=int, default=-1, help='Number of GPUs to use')

    # Preprocessing
    parser.add_argument('--seed', type=int, default=1, help='RNG seed')
    parser.add_argument('--rgb_max', type=float, default=255.0, help='Max RGB value')
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.0,
        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
        help='Spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')

    # Weights
    parser.add_argument('--optical_weights', type=str, help='Path to FlowNet weights', default='')

    ### Model and loss ###
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')
    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

    return parser


def parse_flow_args(parser):
    """
    Process optical flow arguments.

    :param parser: (argparse.ArgumentParser) -> Argument parser that includes flow parser
    :return: (argparse.args) -> Process arguments
    """

    with tools.TimerBlock('Parsing Arguments') as block:
        args, unknown = parser.parse_known_args()
        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()

        # Print all arguments, color the non-defaults
        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.loss_class = tools.module_to_dict(losses)[args.loss]
        args.cuda = torch.cuda.is_available()

    return args


def spatial_parser():
    """
    Create argument parser for spatial CNN class (inference only).

    :return: (argparse.ArgumentParser) -> Parser with arguments for spatial CNN
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--spatial_weights', type=str, help='Path to spatial CNN weights', default='')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Desired input image size')
    return parser


def motion_parser():
    """
    Create argument parser for motion CNN class (inference only).

    :return: (argparse.ArgumentParser) -> Parser with arguments for motion CNN
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--motion_weights', type=str, help='Path to motion CNN weights', default='')
    return parser


def save_buffer(buf, video_path):
    """
    Save frame buffer as video.
    
    :param buf: (np.ndarray) -> List of frames
    :param video_path: (str) -> Path to saved video
    :return: None
    """
    
    rows, cols, channels = buf[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24.0
    sz = (int(cols), int(rows))
    color_frames = True
    writer = cv2.VideoWriter(video_path, fourcc, fps, sz, color_frames)

    for frame in buf:
        if not np.any(frame):
            # Prevent blank frames from being saved to video at beginning of stream
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame.astype(np.uint8))

    writer.release()
