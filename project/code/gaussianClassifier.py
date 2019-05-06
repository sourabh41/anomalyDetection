import torch
from mahalanobisDistance import *

def trainGaussianClassifier(trainX):
	'''
	trainX : All normal extracted regions, N x D
	get a D dimensional gaussian
	'''
	fcn_size = len(trainX) 
	o_size = trainX[0].shape
	dim = 0
	for image in trainX:
		dim = dim + image.shape[0]

	new_trainX = torch.zeros(dim*o_size[2]*o_size[3], o_size[1])
	#print(new_trainX.shape)
	i = 0
	for image in trainX:
		#print("new_trainX",new_trainX[i:image.shape[0]*o_size[2]*o_size[3],:].shape)
		#print("1",image.reshape(image.shape[0]*o_size[2]*o_size[3], o_size[1]).shape)
		#print("2",image.shape)
		#print("3",o_size)
		new_trainX[i:i+image.shape[0]*o_size[2]*o_size[3],:] = image.reshape(image.shape[0]*o_size[2]*o_size[3], o_size[1])
		i += image.shape[0]*o_size[2]*o_size[3]
	del trainX
	#trainX = [image.reshape(o_size[0]*o_size[2]*o_size[3], o_size[1]) for image in trainX]
	mean = torch.mean(new_trainX,dim=0)
	#mean = torch.mean(torch.stack(mean),dim=0)
	
	new_trainX = new_trainX-mean
	#covariance = [torch.matmul(torch.t(image),image)/(image.size(0)-1) for image in X]
	
	#sum_cov = 0
	#for cov in covariance:
		#sum_cov += torch.mul(cov,cov)
	#covariance = torch.sqrt(sum_cov)/(len(covariance)-1)
	#n,d = trainX.size()
	#mean = torch.mean(trainX,dim=0)
	#X = trainX - mean
	n = new_trainX.shape[0]
	covariance = torch.matmul(torch.t(new_trainX),new_trainX)/(n-1)
	mean = torch.t(torch.unsqueeze(mean,dim=1))
	return mean,covariance


if __name__ == "__main__":
	
	trainX = torch.rand(10,20)
	mu,C = trainGaussianClassifier(trainX)

	test = torch.rand(4,5,20,6,8)
	d = mahalanobisDistance(test,mu,C)
	print(d.shape)
