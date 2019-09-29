import argparse
import cv2
import queue

from two_stream_action_recognition.action_recognition import SpatialCNN, MotionCNN
from flownet2_pytorch.optical_flow import OpticalFlow


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

    # CUDA args
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    # FlowNet args
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--rgb_max', type=float, default=255.0)
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                        help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')

    # Model and loss args
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')
    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

    # Weights
    parser.add_argument('--optical_weights', '-ow', type=str, help='path to FlowNet weights')
    parser.add_argument('--spatial_weights', '-sw', type=str, help='path to spatial CNN weights')
    parser.add_argument('--temporal_weights', '-tw', type=str, help='path to motion CNN weights')

    parser.add_argument('--stream', type=str, help='path to video stream')

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
        args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def main():
	args = parse_args()

	cap = cv2.VideoCapture(args.stream)
	optical_flow = OpticalFlow(args)
	spatial_cnn = SpatialCNN(args.spatial_weights)
	motion_cnn = MotionCNN(args.temporal_weights)

	predictions = inference(cap, optical_flow, spatial_cnn, motion_cnn)
	

if __name__ == '__main__':
	main()
