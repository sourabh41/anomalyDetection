import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


class SparseAutoencoder(nn.Module):
	def __init__(self, in_channels, hidden_channels, filter_size = (1,1), stride=1):
		# out_channels = in_channels
		super(SparseAutoencoder, self).__init__()
		self.encoder = nn.Conv2d(in_channels, hidden_channels, filter_size, stride)
		self.decoder = nn.Conv2d(hidden_channels, in_channels, filter_size, stride)

	def forward_train(self, x):
		# x : output of 2nd convolution layers
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)

		return encoded, decoded

	def forward(self,x):
		encoded = self.encoder(x.view(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
		return encoded.view(x.shape[0],x.shape[1],encoded.shape[1],encoded.shape[2],encoded.shape[3])

	

	def train(self,optimizer,train_data,epochs,batch_size,rho,gamma=3):
		num_videos = train_data.shape[0]
		num_frames = train_data.shape[1]

		train_loader = torch.utils.data.DataLoader(
                dataset=train_data.view(num_frames*num_videos,train_data.shape[2],train_data.shape[3],train_data.shape[4]),
                batch_size=batch_size,
        shuffle=True)
	
		for epoch in range(epochs):
			print("Epoch :", epoch)
			for batch_num, x in enumerate(train_loader):
				x = Variable(x)
				encoded, decoded = self.forward_train(x)
				MSE_loss = (x - decoded) ** 2
				MSE_loss = MSE_loss.view(1, -1).sum(1)/batch_size
				rho_hat = torch.mean(torch.sigmoid(encoded), dim=0)
				sparsity_penalty = gamma * sparsity_loss(rho_hat, rho)
				loss = MSE_loss + sparsity_penalty
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			print("Loss :",loss.data[0].item(), "MSE Loss :", MSE_loss[0].item())

		


def sparsity_loss(rho_hat, rho):
	
		tensor = torch.ones(rho_hat.shape)
		s1 = torch.sum(rho*torch.log(rho*tensor/rho_hat))
		s2 = torch.sum((1-rho)*torch.log((1-rho)*tensor/(tensor-rho_hat)))
		return s1 + s2