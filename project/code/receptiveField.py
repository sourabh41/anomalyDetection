import torch


# Loop this function over all layers in our network to get the anomaly region


def receptiveField(inputSize,filterSize,stride,outputPosition,padding):
	'''
	Example Input:
	inputSize = [200,300]		Tensor
	filterSize = [10,10]		Tensor
	stride = 1
	outputPosition = [[1,2]] Tensor (one point)  1 is row and 2 is col
				   = [[1,2],[4,6]]		Tensor  (square of output), [1,2] is top left, [4,6] is bottom right

	outputs a square of input which affects output positions  
	Format :	[[1,2],[4,6]]		Tensor  (square of output), [1,2] is top left, [4,6] is bottom right	 

	'''

	assert(len(inputSize) == 2)
	assert(len(filterSize) == 2)
	assert(len(outputPosition) == 1 or len(outputPosition) == 2)
	assert(len(outputPosition[0] == 2))

	if(len(outputPosition) == 1):
		row = outputPosition[0,0]
		col = outputPosition[0,1]
		top_left_corner = [row*stride-padding,col*stride-padding]
		bottom_right_corner = [row*stride+filterSize[0]-1-padding,col*stride+filterSize[1]-1-padding]
		
		top_left_corner[0] = max(0,top_left_corner[0])
		top_left_corner[1] = max(0,top_left_corner[1])
		
		bottom_right_corner[0] = min(inputSize[0]-1,bottom_right_corner[0])
		bottom_right_corner[1] = min(inputSize[1]-1,bottom_right_corner[1])

		return torch.Tensor([top_left_corner,bottom_right_corner])
	
	elif(len(outputPosition) == 2):
		
		top_left = receptiveField(inputSize,filterSize,stride,torch.unsqueeze(outputPosition[0],0),padding)
		bottom_right = receptiveField(inputSize,filterSize,stride,torch.unsqueeze(outputPosition[1],0),padding)
		output = torch.ones((2,2))
		output[0] = top_left[0]
		output[1] = bottom_right[1]
		return output


def getRegionfromAlex(outputPosition):

	'''
	outputs a square of input which affects output positions  
	Format :	[[1,2],[4,6]]		Tensor  (square of output), [1,2] is top left, [4,6] is bottom right
				or [[1,2]]			Tensor
	'''
	region = receptiveField(torch.Tensor([27,27]),torch.Tensor([5,5]),1,outputPosition,2)
	region = receptiveField(torch.Tensor([55,55]),torch.Tensor([3,3]),2,region,0)
	region = receptiveField(torch.Tensor([224,224]),torch.Tensor([11,11]),4,region,2)

	return region

if __name__ == "__main__":
	field = receptiveField(torch.Tensor([6,6]),torch.Tensor([2,2]),2,torch.Tensor([[0,0],[2,2]]),0)
	print(field)