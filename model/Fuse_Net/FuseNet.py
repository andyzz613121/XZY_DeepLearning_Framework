from model.SegNet_prio_success import SegNet_img
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from copy import deepcopy

import sys
base_path = 'D:\\Code\\LULC\\Laplace\\'
sys.path.append(base_path)
from model import SegNet

class Fusenet(nn.Module):
	def __init__(self, RGB_input_channel, Guiding_input_channel, num_labels):
		super(Fusenet, self).__init__()
		if RGB_input_channel <= 3:
			RGB_branch = SegNet.SegNet(RGB_input_channel, num_labels).cuda()
		elif RGB_input_channel > 3:
			RGB_branch_pre = SegNet.SegNet(3, num_labels).cuda()
			RGB_branch = SegNet.SegNet(RGB_input_channel, num_labels).cuda()
			RGB_branch = SegNet.add_conv_channels(RGB_branch, RGB_branch_pre, [RGB_input_channel-3])

		if Guiding_input_channel <= 3:
			Guiding_branch = SegNet.SegNet(Guiding_input_channel, num_labels).cuda()
		elif Guiding_input_channel > 3:
			Guiding_branch_pre = SegNet.SegNet(3, num_labels).cuda()
			Guiding_branch = SegNet.SegNet(Guiding_input_channel, num_labels).cuda()
			Guiding_branch = SegNet.add_conv_channels(Guiding_branch, Guiding_branch_pre, [Guiding_input_channel-3])

		self.CBR1_DEPTH_ENC = Guiding_branch.conv_1
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
		self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.dropout = nn.Dropout(p=0.4)
		self.CBR2_DEPTH_ENC = Guiding_branch.conv_2
		self.CBR3_DEPTH_ENC = Guiding_branch.conv_3
		self.CBR4_DEPTH_ENC = Guiding_branch.conv_4
		self.CBR5_DEPTH_ENC = Guiding_branch.conv_5

		self.CBR1_RGB_ENC = RGB_branch.conv_1
		self.CBR2_RGB_ENC = RGB_branch.conv_2
		self.CBR3_RGB_ENC = RGB_branch.conv_3
		self.CBR4_RGB_ENC = RGB_branch.conv_4
		self.CBR5_RGB_ENC = RGB_branch.conv_5

		self.CBR1_RGB_DEC = RGB_branch.deconv_1
		self.CBR2_RGB_DEC = RGB_branch.deconv_2
		self.CBR3_RGB_DEC = RGB_branch.deconv_3
		self.CBR4_RGB_DEC = RGB_branch.deconv_4
		self.CBR5_RGB_DEC = RGB_branch.deconv_5

	def forward(self, RGB_input, Guiding_input):
		########  DEPTH ENCODER  ########
		# Stage 1
		x_1 = self.CBR1_DEPTH_ENC(Guiding_input)
		x, id1_d = self.pool(x_1)

		# Stage 2
		x_2 = self.CBR2_DEPTH_ENC(x)
		x, id2_d = self.pool(x_2)

		# Stage 3
		x_3 = self.CBR3_DEPTH_ENC(x)
		x, id3_d = self.pool(x_3)
		x = self.dropout(x)

		# Stage 4
		x_4 = self.CBR4_DEPTH_ENC(x)
		x, id4_d = self.pool(x_4)
		x = self.dropout(x)

		# Stage 5
		x_5 = self.CBR5_DEPTH_ENC(x)

		########  RGB ENCODER  ########

		# Stage 1
		y = self.CBR1_RGB_ENC(RGB_input)
		y = torch.add(y,x_1)
		y = torch.div(y,2)
		y, id1 = self.pool(y)

		# Stage 2
		y = self.CBR2_RGB_ENC(y)
		y = torch.add(y,x_2)
		y = torch.div(y,2)
		y, id2 = self.pool(y)

		# Stage 3
		y = self.CBR3_RGB_ENC(y)
		y = torch.add(y,x_3)
		y = torch.div(y,2)
		y, id3 = self.pool(y)
		y = self.dropout(y)

		# Stage 4
		y = self.CBR4_RGB_ENC(y)
		y = torch.add(y,x_4)
		y = torch.div(y,2)
		y, id4 = self.pool(y)
		y = self.dropout(y)

		# Stage 5
		y = self.CBR5_RGB_ENC(y)
		y = torch.add(y,x_5)
		y = torch.div(y,2)
		y_size = y.size()

		y, id5 = self.pool(y)
		y = self.dropout(y)

		########  DECODER  ########

		# Stage 5 dec
		y = self.unpool(y, id5,output_size=y_size)
		y = self.CBR5_RGB_DEC(y)
		y = self.dropout(y)

		# Stage 4 dec
		y = self.unpool(y, id4)
		y = self.CBR4_RGB_DEC(y)
		y = self.dropout(y)

		# Stage 3 dec
		y = self.unpool(y, id3)
		y = self.CBR3_RGB_DEC(y)
		y = self.dropout(y)

		# Stage 2 dec
		y = self.unpool(y, id2)
		y = self.CBR2_RGB_DEC(y)

		# Stage 1 dec
		y = self.unpool(y, id1)
		y = self.CBR1_RGB_DEC(y)

		return y
