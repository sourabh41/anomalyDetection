from PIL import Image
import PIL
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
import os, os.path
import glob
import pickle
# import cv2

import numpy

OUTPUT_PATH = "./output/"
DATA_PATHS = dict()
DATASETS_DIR = "../datasets/prepared/"
DATA_PATHS["sample_uscd1seq"] = "../datasets/ucsd/UCSDped1/Test/"#,"uscd2seq"]#,"mall1seq","mall2seq","mall3seq","sub_entry","sub_exit"]
DATA_PATHS["uscd1seq"] = "../datasets/ucsd/UCSDped1/Test/"


def saveAbnormalRegions(abnormal_frames, abnormal_positions, abnormal_regions, dataset_name, save_path_name):
    
    
    if not os.path.exists(save_path_name):
        os.makedirs(save_path_name)
    fetch_path = DATA_PATHS[dataset_name]
    
    number_of_directories = len(abnormal_frames)
    dir_prefix = "Test"
    
    """for name in os.listdir(path):
                    if os.path.isdir(os.path.join(path, name)):
                        if (name[-1] == "t"):
                            continue
                        number_of_directories += 1 """


    for dir_no in range(number_of_directories):
        if ((torch.sum(abnormal_frames[dir_no])).item() == 0):
            continue
        n_detectable_frames = (abnormal_frames[dir_no].shape)[0]
        dir_path = fetch_path + "Test"+str(dir_no+1).zfill(3) + "/"
        for frame_no in range(n_detectable_frames):
            if (abnormal_frames[dir_no][frame_no] == 0):
                continue
            frame_path = dir_path + str(frame_no+1).zfill(3) + ".tif"
            image = Image.open(frame_path)
            
            image = image.resize((224, 224), Image.ANTIALIAS)
            pixels = image.load()
            #print(abnormal_regions[(dir_no,frame_no)].shape)
            n_squares = len(abnormal_regions[(dir_no,frame_no)])
            for square in range(n_squares):
                x1 = min(int(abnormal_regions[(dir_no,frame_no)][square][0,0]),223-7)
                x2 = min(int(abnormal_regions[(dir_no,frame_no)][square][1,0]),223)
                y1 = min(int(abnormal_regions[(dir_no,frame_no)][square][0,1]),223-7)
                y2 = min(int(abnormal_regions[(dir_no,frame_no)][square][1,1]),223-7)
                #print(pixels[x1,y1])
                for x in range(x1,x1+7):
                    pixels[x,y1] = 255
                for y in range(y1,y1+7):
                    pixels[x1,y] = 255
                for x in range(x2-7,x2):
                    pixels[x,y2] = 255
                for y in range(y2-7,y2):
                    pixels[x2,y] = 255

            image.save(save_path_name+"/"+str(dir_no+1).zfill(3)+"_"+str(frame_no+1).zfill(3)+".tif")

            



    return

def calculateAccuracy(abnormal_frames, dataset_name):
    labels_path = DATASETS_DIR + dataset_name + "_labels_test.pkl"

    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)

    accuracy = []
    total_frames = 0
    for i in range(len(abnormal_frames)):
        accuracy.append(torch.sum(torch.eq(labels[i].int(), abnormal_frames[i].int())))
        total_frames += (labels[i].shape)[0]
    


    return float(sum(accuracy)*100)/(1.0*total_frames)
