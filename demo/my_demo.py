import sys
import argparse
import cv2

import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.strided_transformer import Model
from common.camera import *



def pose_lift(keypoints, video_path, pretrained_path, frames):
	args, _ = argparse.ArgumentParser().parse_known_args()
	args.layers, args.channel, args.d_hid, args.frames = 3, 256, 512, frames
	args.stride_num = [3, 9, 13]
	args.pad = (args.frames - 1) // 2

	args.n_joints, args.out_joints = 17, 17

	# load model

	## Reload 
	model = Model(args).cuda()
	model_dict = model.state_dict()

	model_path = pretrained_path
	pre_dict = torch.load(model_path)
	for name, key in model_dict.items():
		# print('name / key: ', name, key )

		model_dict[name] = pre_dict[name]
	model.load_state_dict(model_dict)

	model.eval()

	## load 2d keypoints as input
	keypoints = np.expand_dims(keypoints, 0)
	print('keypoints shape: ', keypoints.shape)

	# valid_frames = keypoints.shape[0]

	# load input video
	cap = cv2.VideoCapture(video_path)
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print('width: ', width)
	print('height: ', height)
	print('video_length: ', video_length)
	## 3D
	print('\nGenerating 3D pose...')

	def gen_pose(args, model, keypoints, width, height):
		output_3d = []
		frames_2d = keypoints.shape[1]  # (1, frames, 17,2)

		for i in tqdm(range(frames_2d)):
			start = max(0, i - args.pad) 
			end =  min(i + args.pad, len(keypoints[0])-1)

			input_2D_no = keypoints[0][start:end+1]

			left_pad, right_pad = 0, 0
			if input_2D_no.shape[0] != args.frames:
				if i < args.pad:
					left_pad = args.pad - i
				if i > len(keypoints[0]) - args.pad - 1:
					right_pad = i + args.pad - (len(keypoints[0]) - 1)

				input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')

			joints_left =  [4, 5, 6, 11, 12, 13]
			joints_right = [1, 2, 3, 14, 15, 16]
			# input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  
			input_2D = normalize_screen_coordinates(input_2D_no, w=height, h=width) 
			flip_ = False
			

			input_2D_aug = copy.deepcopy(input_2D)
			input_2D_aug[ :, :, 0] *= -1
			input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
			input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
			
			input_2D = input_2D[np.newaxis, :, :, :, :]

			input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

			N = input_2D.size(0)

			## estimation
			output_3D_non_flip, _ = model(input_2D[:, 0])
			output_3D_flip, _     = model(input_2D[:, 1])

			output_3D_flip[:, :, :, 0] *= -1
			output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 
			output_3D = (output_3D_non_flip + output_3D_flip) / 2

			output_3D[:, :, 0, :] = 0
			post_out = output_3D[0, 0].cpu().detach().numpy()

			rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
			
			rot = np.array(rot, dtype='float32')
			post_out = camera_to_world(post_out, R=rot, t=0)

			post_out[:, 2] -= np.min(post_out[:, 2])
			
			output_3d.append(post_out)
		return np.array(output_3d)
	prediction_3d = gen_pose(args, model, keypoints, width, height)
	print('\nFinished generating 3D pose...')
	return prediction_3d
