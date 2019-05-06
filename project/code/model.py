import torch
import torch.nn as nn
import torchvision
from torchvision import models


class myModel(nn.Module):
	def __init__(self):
		super(myModel,self).__init__()
		alexnet = models.alexnet(pretrained=True)
		self.conv1 = alexnet.features[0]
		self.relu1 = alexnet.features[1]
		self.maxpool1 = alexnet.features[2]
		self.conv2 = alexnet.features[3]
		self.relu2 = alexnet.features[4]
	
	def forward(self,x):
		out1 = self.conv1(x)
		out1 = self.relu1(out1)
		out1 = self.maxpool1(out1)
		out1 = self.conv2(out1)
		out1 = self.relu2(out1)
		
		return out1

	def getOutputs(self,x):

		'''
		Get outputs for all videos and frames
		'''
		#FCNoutputs = torch.zeros(x.shape[0], x.shape[1], 192, 27, 27)
		FCNoutputs = [torch.zeros(image.shape[0], 192, 27, 27) for image in x]

		"""for v in range(x.shape[0]):
			inp = x[v,:,:,0:224,0:224]
			output = self.forward(inp)
			FCNoutputs[v,:] = output"""
		for v in range(len(x)):
			inp = x[v][:,:,0:224,0:224]
			#print("here", inp.shape)
			output = self.forward(inp)
			FCNoutputs[v][:] = output

		return FCNoutputs
