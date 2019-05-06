import torch

def mahalanobisDistance(x,mean,covariance):
	'''
	Input
	x = D(#depth) x width x height
	mean = 1xD
	C = DxD

	Output = width x height
	'''

	x = x.permute(1,2,0)
	mean_sub_x = x - mean
	size = x.shape
	product = torch.matmul(
		torch.matmul(mean_sub_x,torch.inverse(covariance)).contiguous().view(size[0]*size[1],size[2]),
		mean_sub_x.contiguous().view(size[2],size[0]*size[1]))
	#print(x.shape)
	#print(torch.sqrt(torch.diagonal(product).reshape(size[0],size[1])).shape)
	return torch.sqrt(torch.diagonal(product).reshape(size[0],size[1]))