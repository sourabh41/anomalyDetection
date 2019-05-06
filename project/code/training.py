import torch
import torch.nn as nn
import torchvision
import pickle
from torchvision import models
from model import *
from gaussianClassifier import *
from receptiveField import *
from mahalanobisDistance import *
from output import *
from datetime import datetime
from sparseAutoEncoder import *
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

#   DATA PATH
#######################################################################################################

DATASETS_DIR = "../datasets/prepared/"
DATA_LIST = ["sample_uscd1seq"]#,"mall1seq","mall2seq","mall3seq","sub_entry","sub_exit"] #uncomment when using


#######################################################################################################
#   HYPERPARAMETERS

# To be tuned

epochs = 10
batch_size = 32
rho = 0.05
gamma = 3
lr = 1e-3
phi = 5
alpha = 10
beta = 5

# CREATE MODEL INSTANCE 
#######################################################s################################################
def train():
	model_ft = myModel()
	mean_cov_dict = dict()

	

	for dataset_name in DATA_LIST:
		print("Loading Training Data ", dataset_name)
		with open(DATASETS_DIR+dataset_name+"_train.pkl", 'rb') as f:
			train_data = pickle.load(f)

		print("Training SparseAutoencoder")
		FCNtrain_outputs = model_ft.getOutputs(train_data)

		sparse_auto_encoder = SparseAutoencoder(192, 50)
		optimizer = optim.Adam(sparse_auto_encoder.parameters(), lr=lr)
		sparse_auto_encoder.train(optimizer,torch.stack(FCNtrain_outputs),epochs,batch_size,rho,gamma)

		# train another gaussian classifier on these sparse_auto_encoder_output for suspicious regions
		sparse_auto_encoder_outputs = sparse_auto_encoder.forward(torch.stack(FCNtrain_outputs))


		print("Training GaussianClassifier")
		mean_cov_dict[dataset_name] = [trainGaussianClassifier(FCNtrain_outputs), trainGaussianClassifier(list(sparse_auto_encoder_outputs))]

	return mean_cov_dict , model_ft, sparse_auto_encoder


def test(mean_cov_dict, model_ft, sparse_auto_encoder,alpha, beta, phi):
	
	mean_cov_dict["meta_data"] = dict()
	mean_cov_dict["meta_data"]["alpha"] = alpha
	mean_cov_dict["meta_data"]["beta"] = beta
	for dataset_name in DATA_LIST:
		print("Loading Testing Data ", dataset_name)
		with open(DATASETS_DIR+dataset_name+"_test.pkl", 'rb') as f:
			test_data = pickle.load(f)

		print("Testing")
		FCNtest_outputs = model_ft.getOutputs(test_data)
		sparse_auto_encoder_outputs = sparse_auto_encoder.forward(torch.stack(FCNtest_outputs))

		fcn_size = len(FCNtest_outputs) 
		
		#abnormal_frames = [0 for i in range(fcn_size)]
		abnormal_frames = []
		abnormal_positions = [torch.zeros(test_frame.shape[0],23,23) for test_frame in test_data]
		abnormal_regions = {}
		[meanG1, covarianceG1] = mean_cov_dict[dataset_name][0]
		[meanG2, covarianceG2] = mean_cov_dict[dataset_name][1]
		for vid in range(fcn_size):
			print(vid+1, "/",fcn_size)
			test_frame = test_data[vid]
			abnormal_frames.append(torch.zeros(test_frame.shape[0]).int())
			for frame in range(test_frame.shape[0]):
				abnormal_regions[(vid,frame)] = []
				frame_output = FCNtest_outputs[vid][frame]
				distances = mahalanobisDistance(frame_output,meanG1,covarianceG1)
				# print(frame_output.shape)
				for i in range(23):
					for j in range(23):
						if(distances[i,j] > alpha):
							abnormal_frames[vid][frame] = 1
							abnormal_positions[vid][frame,i,j] = 1
							abnormal_regions[(vid,frame)].append(getRegionfromAlex(torch.Tensor([[i,j]])))
						elif(distances[i,j] > beta):
							dist = mahalanobisDistance(torch.unsqueeze(torch.unsqueeze(sparse_auto_encoder_outputs[vid,frame,:,i,j],1),2),meanG2,covarianceG2)
							if(dist > phi):
								abnormal_frames[vid][frame] = 1
								abnormal_positions[vid][frame,i,j] = 1
								abnormal_regions[(vid,frame)].append(getRegionfromAlex(torch.Tensor([[i,j]])))



		#print(abnormal_regions[(0,0)])

		#print(abnormal_regions[(5,5)])
		#speed this up later

		accuracy = calculateAccuracy(abnormal_frames, dataset_name)
		now = datetime.now()
		save_path_name = OUTPUT_PATH+str(now.strftime("%Y%m%d"))+"_"+str(alpha).zfill(3)+"_"+str(beta).zfill(3)
		# print("Saving Regions")
		# saveAbnormalRegions(abnormal_frames, abnormal_positions, abnormal_regions, dataset_name, save_path_name)
	return accuracy


def main():
	mean_cov_dict , model_ft, sparse_auto_encoder = train( )
	"""for alpha in range(242,260,2):
					accuracy = test(mean_cov_dict = mean_cov_dict, model_ft = model_ft, alpha = alpha*1.0/10, beta = 5, )
					print(alpha*1.0/10, accuracy)"""
	#optimum alpha 25.2 accuracy - 67.49%
	accuracy = test(mean_cov_dict = mean_cov_dict, model_ft = model_ft, sparse_auto_encoder = sparse_auto_encoder, alpha = alpha, beta = beta, phi = phi)
	print("Accuracy obtained is" , accuracy)

if __name__ == '__main__':
	main()