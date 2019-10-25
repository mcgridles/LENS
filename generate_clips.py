import os
import sys
import cv2
import argparse
import numpy as np
import torch
import colorama

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
	assert len(video) == len(flow[0])
	assert len(video) == len(flow[1])

	video_name = os.path.splitext(os.path.basename(video_path))[0]
	chunks = int(video.shape[0] / (fps * duration))

	rgb_dir = os.path.join(output_dir, 'rgb')
	ofu_dir = os.path.join(output_dir, 'flownet2', 'u')
	ofv_dir = os.path.join(output_dir, 'flownet2', 'v')

	save_clips(rgb, video_name, rgb_dir)
	save_clips(flow[0], video_name, of_u_dir)
	save_clips(flow[1], video_name, of_v_dir)


def save_clips(chunks, video, video_name, output_dir):
	"""
	Save clip chunks from 
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
			frame = clip[frame_num, :, :, :]
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


def optical_flow(of, video_path):
	cap = cv2.VideoCapture(video_path)

	u = [], v = []
	previous_frame = None
	ret = True
	while ret: 
		ret, frame = cap.read()

		if previous_frame is not None:
			flow = optical_flow.run([prev_frame, frame])
			u.append(flow[:, :, 0])
			v.append(flow[:, :, 1])
	
	u = np.array(u)
	v = np.array(v)
	return u, v


def parse_args():
	parser = argparse.ArgumentParser('Clip Generator')
	flow = parser.add_argument_group('optical flow')

	# CUDA
	flow.add_argument('--number_gpus', '-ng', type=int, default=-1, help='Number of GPUs to use')

	# Preprocessing
	flow.add_argument('--seed', type=int, default=1, help='RNG seed')
	flow.add_argument('--rgb_max', type=float, default=255.0, help='Max RGB value')
	flow.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
	flow.add_argument('--fp16_scale', type=float, default=1024.0,
		help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
	flow.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
		help='Spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')

	# Weights
	flow.add_argument('--optical_weights', '-ow', type=str, help='Path to FlowNet weights', default='')

	# Model and loss
	tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')
	tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

	# Clip generation
	parser.add_argument('--video', '-v', help='Path to input video', type=str, required=True)
	parser.add_argument('--output', '-o', help='Path to output directory', type=str, required=True)
	parser.add_argument('--duration', '-d', help='Duration of each clip in seconds', type=int, default=4)

	with tools.TimerBlock('Parsing Arguments') as block:
		args, unknown = parser.parse_known_args()
		if args.number_gpus < 0:
			args.number_gpus = torch.cuda.device_count()

		# Have to do it this way since there seem to be issues using `required=True` in `add_argument()`
		if not args.optical_weights:
			raise Exception('Weights are required')

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


def main():
	"""
	Example:
	>>> python generate_clips -v /path/to/video.mov -o /path/to/directory -d 3
	"""

	args = parse_args()
	of = OpticalFlow(args)
	flow = optical_flow(of, args.video)

	generate_clips(args.video, args.output, args.duration, flow)


if __name__ == '__main__':
	main()
