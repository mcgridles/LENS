import os
import sys
import cv2
import argparse
import numpy as np
import glob
from utils import flow_parser, parse_flow_args

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'flownet2-pytorch'))
from optical_flow import OpticalFlow
from optical_flow import models, losses, tools


def generate_clips(video_path, output_dir, duration, flow):
    """
    Generate random clips from video file.

    :param video_path: (str) -> Path to the video to be split
    :param output_dir: (str) -> Path to directory to save clips in
    :param duration: (int) -> Duration of the new split videos in seconds
    :param flow: (np.ndarray) -> U and V channels of the optical flow calculated over the video
    """

    rgb, fps = load_video(video_path)
    assert len(rgb) == len(flow[0])
    assert len(rgb) == len(flow[1])

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    chunks = int(rgb.shape[0] / (fps * duration))

    rgb_dir = os.path.join(output_dir, 'rgb')
    of_u_dir = os.path.join(output_dir, 'flownet2', 'u')
    of_v_dir = os.path.join(output_dir, 'flownet2', 'v')

    chunk_and_save(rgb, chunks, video_name, rgb_dir)
    chunk_and_save(flow[0], chunks, video_name, of_u_dir)
    chunk_and_save(flow[1], chunks, video_name, of_v_dir)


def chunk_and_save(video, chunks, video_name, output_dir):
    """
    Separates video into chunks and saves the frames.

    :param video: (np.ndarray) -> Array of frames from video
    :param chunks: (int) -> Number of chunks to divide video into
    :param video_name: (str) -> Name of video file
    :param output_dir: (str) -> Path to output directory
    """

    clips = np.array_split(video, chunks, 0)
    for i, clip in enumerate(clips):
        clip_num = '_c{}'.format(str(i+1).zfill(6))
        clip_name = video_name + clip_num

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
    >>> python generate_clips -v /path/to/video.mov -o /path/to/directory -d 3
    """

    args = parse_args()
    of = OpticalFlow(args)
    flow = generate_flow(of, args.video)

    video_files = glob.glob(os.path.join(args.video, '*{}'.format(args.ext)))
    for video in video_files:
        generate_clips(video, args.output, args.duration, flow)


if __name__ == '__main__':
    main()
