import os
import cv2
import argparse
import numpy as np


def generate_clips(video_path, output_dir, duration):
    """
    Generate random clips from video file.

    :param video_path: (str) -> Path to the video to be split
    :param output_dir: (str) -> Path to directory to save clips in
    :param duration: (int) -> Duration of the new split videos in seconds
    """

    video, fps = load_video(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    chunks = int(video.shape[-1] / (fps * duration))
    sub_videos = np.array_split(video, chunks, 3)
    
    for i, vid in enumerate(sub_videos):
    	clip_num = '_c{}'.format(str(i+1).zfill(10))
        clip_name = video_name + clip_num

        # Create directory if it doesn't exist
        dir_name = os.path.join(output_dir, clip_name)
	    if not os.path.exists(dir_name):
	    	os.makedirs(dir_name)
	    	print('Creating directory: {}'.format(dir_name))

	    # Save frames
        for frame_num in range(vid.shape[-1]):
            frame = vid[..., frame_num]
            clip_path = os.path.join(output_dir, 
            						 clip_name, 
            						 'frame{}.jpg'.format(str(frame_num+1).zfill(6))))
            cv2.imwrite(clip_path, frame)


def load_video(video_path):
	"""
	Load video frames into array.

	:param video_path: (str) -> Path to video file
	:return: (list(frame)) -> All frames from the video
			 (float) -> Video framerate
	"""

    cap = cv2.VideoCapture(video_path) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video = []
    ret = True
    while ret: 
        ret, frame = cap.read()

        if frame is not None:
            video.append(frame)
     
    video = np.stack(video, 3)
    return video, fps


def main():
	"""
	Example:
	>>> python generate_clips -v /path/to/video.mov -o /path/to/directory -d 3
	"""

	parser = argparse.ArgumentParser('Clip Generator')
	parser.add_argument('--video', '-v', help='Path to input video', type=str, required=True)
	parser.add_argument('--output', '-o', help='Path to output directory', type=str, required=True)
	parser.add_argument('--duration', '-d', help='Duration of each clip in seconds', type=int, default=4)
	args = parser.parse_args()

	generate_clips(args.video, args.output, args.duration)


if __name__ == '__main__':
	main()
