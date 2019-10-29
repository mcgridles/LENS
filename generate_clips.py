import os
import sys
import cv2
import argparse
import numpy as np
import glob
from collections import defaultdict
from utils import flow_parser, parse_flow_args

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'flownet2-pytorch'))
from optical_flow import OpticalFlow
from optical_flow import models, losses, tools


def generate_clips(video_name, video_path, output_dir, duration, flow, start_idx=0):
    """
    Generate random clips from video file.

    :param video_name: (str) -> Base name of video
    :param video_path: (str) -> Path to the video to be split
    :param output_dir: (str) -> Path to directory to save clips in
    :param duration: (int) -> Duration of the new split videos in seconds
    :param flow: (np.ndarray) -> U and V channels of the optical flow calculated over the video
    :param start_idx: (int) -> Starting index for clip numbering
    :return: (int) -> Highest clip index for the current action/group configuration
    """

    rgb, fps = load_video(video_path)
    assert len(rgb) == len(flow[0])
    assert len(rgb) == len(flow[1])

    chunks = int(rgb.shape[0] / (fps * duration))

    rgb_dir = os.path.join(output_dir, 'rgb')
    of_u_dir = os.path.join(output_dir, 'flownet2', 'u')
    of_v_dir = os.path.join(output_dir, 'flownet2', 'v')

    max_clip_idx = chunk_and_save(rgb, chunks, video_name, rgb_dir)
    chunk_and_save(flow[0], chunks, video_name, of_u_dir)
    chunk_and_save(flow[1], chunks, video_name, of_v_dir)

    return max_clip_idx


def chunk_and_save(video, chunks, video_name, output_dir, start_idx=0):
    """
    Separates video into chunks and saves the frames.

    :param video: (np.ndarray) -> Array of frames from video
    :param chunks: (int) -> Number of chunks to divide video into
    :param video_name: (str) -> Name of video file
    :param output_dir: (str) -> Path to output directory
    :param start_idx: (int) -> Starting index for clip numbering
    :return: (int) -> Highest clip index for the current action/group configuration
    """

    clips = np.array_split(video, chunks, 0)
    max_clip_idx = 0
    for i, clip in enumerate(clips):
        clip_num = '_c{}'.format(str(i + start_idx).zfill(6))
        clip_name = video_name + clip_num
        max_clip_idx = i + start_idx

        # Create directory if it doesn't exist
        dir_name = os.path.join(output_dir, clip_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('Creating directory: {}'.format(dir_name))

        # Save frames
        for frame_num in range(clip.shape[0]):
            try:
                frame = clip[frame_num, :, :, :]
            except IndexError:
                frame = clip[frame_num, :, :]

            clip_path = os.path.join(output_dir, 
                                     clip_name, 
                                     'frame{}.jpg'.format(str(frame_num+1).zfill(6)))
            cv2.imwrite(clip_path, frame)

    return max_clip_idx


def load_video(video_path):
    """
    Load video frames into array.

    :param video_path: (str) -> Path to video file
    :return: (list(frame)) -> All frames from the video
             (float) -> Video framerate
    """

    print('Loading video')
    cap = cv2.VideoCapture(video_path) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video = []
    ret = True
    while ret: 
        ret, frame = cap.read()

        if frame is not None:
            video.append(frame)
    
    video = np.array(video[:-1])

    return video, fps


def generate_flow(of, video_path):
    """
    Generate optical flow frames from video.

    :param of: (OpticalFlow) -> Flownet2 wrapper object for calculating optical flow
    :param video_path: (str) -> Path to video file
    :return: (np.ndarray) -> x component of the optical flow
             (np.ndarray) -> y component of the optical flow
    """

    print('Generating optical flow')
    cap = cv2.VideoCapture(video_path)

    u, v = [], []
    prev_frame = None
    ret = True
    while ret: 
        ret, frame = cap.read()

        if prev_frame is not None and frame is not None:
            flow = of.run([prev_frame, frame])
            u.append(flow[0, :, :])
            v.append(flow[1, :, :])

        prev_frame = frame
    
    u = np.array(u)
    v = np.array(v)
    return u, v


def parse_args():
    """
    Parse command line arguments.

    :return: (argparse.args) -> Argument object
    """

    parser = argparse.ArgumentParser('Clip Generator', parents=[flow_parser()])

    # Clip generation
    parser.add_argument('--video', '-v', help='Path to directory containing videos', type=str)
    parser.add_argument('--output', '-o', help='Path to output directory', type=str)
    parser.add_argument('--duration', '-d', help='Duration of each clip in seconds', type=int, default=4)
    parser.add_argument('--ext', help='Video file extension', type=str, default='.mov')

    args = parse_flow_args(parser)

    return args


def main():
    """
    Example:
    >>> python generate_clips -v /path/to/video/directory \
                              -o /path/to/output/directory \
                              -ow /path/to/optical/weights.pth.tar \
                              -d 3

    Video naming scheme: v_<ACTION>_g<XX>_v<Y>_<Z>
        ACTION = Name of action
        XX     = Two digit group number
        Y      = Clip number (for when multiple videos of same group)
        Z      = Letter denoting camera view (i.e. a, b, c, etc.)
    """

    args = parse_args()
    of = OpticalFlow(args)

    video_record = defaultdict(int)
    video_files = glob.glob(os.path.join(args.video, '*{}'.format(args.ext)))
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_name = video_name[:-5]
        start_idx = video_record[video_name] + 1

        flow = generate_flow(of, video_path)
        max_clip_idx = generate_clips(video_name, video_path, args.output, args.duration, flow, start_idx)
        
        video_record[video_name] = max_clip_idx


if __name__ == '__main__':
    main()
