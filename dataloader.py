import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import json

from PIL import Image

import os
import argparse
import numpy as np
class Dataset_Interpreter(data.Datase):
	def __init__(self, data_path, file_names, labels=None, transforms=None):
		self.data_path = data_path
		self.file_names = file_names
		self.labels = labels
		self.transforms = transforms

	def __len__(self):
		return (len(self.file_names))

	def __getitem__(self, idx):
		img_name = f'{self.file_names.iloc[idx]}.jpg'
		full_address = os.path.join(self.data_path, img_name)
		image = Image.open(full_address)
		label = self.labels.iloc[idx]

		if self.transforms is not None:
			image = self.transforms(image)

		return np.array(image), label
# class PoseDataset(data.Dataset):
# 	""" Pose custom dataset compatible with torch.utils.data.DataLoader. """
# 	def __init__(self, annotation, imroot, hroot, oproot, vocab, seq_length, transform=None):
# 		self.root = root
# 		self.image_dir = image_dir
# 		self.image_files = os.listdir(image_dir)
# 		self.data = pd.read_csv(csv_file).iloc[:, 1]
# 		self.transform = transform
#
#
# 	def __getitem__(self, index):
# 		imroot = self.imroot
# 		hroot = self.hroot
# 		oproot = self.oproot
# 		vocab = self.vocab
# 		annotation = self.annotation
# 		path, end = annotation.anns[index]
# 		images = []
# 		poses = []
# 		poses2 = []
# 		homography = []
# 		for i in range (end-self.seq_length, end):
# 			image, upp_pose, low_pose, h, pose2 = getPair(imroot, hroot, oproot, path, vocab, i)
# 			if self.transform is not None:
# 				image = self.transform(image)
# 			poses.append((upp_pose, low_pose))
# 			images.append(image)
# 			homography.append(h)
# 			poses2.append(pose2)
# 		homography = torch.Tensor(homography)
# 		images = torch.stack(images)
# 		target = torch.Tensor(poses)
# 		poses2 = torch.Tensor(poses2)
# 		return images, target, homography, poses2
#
#
# 	def __len__(self):
# 		return len(self.annotation)

