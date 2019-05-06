from PIL import Image
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
import os, os.path
import glob
import pickle
import cv2

PREPARED_PATH = "../datasets/prepared/" 
USCD_PATH = "../datasets/ucsd/"
SUB_PATH = "../datasets/subway/"

USCD1_DIRNAME = "UCSDped1"
USCD2_DIRNAME = "UCSDped2"
SUBENTRY_DIRNAME = "sub_entry"
SUBEXIT_DIRNAME = "sub_exit"
MALL1_DIRNAME = "mall1"
MALL2_DIRNAME = "mall2"
MALL3_DIRNAME = "mall3"

def generateTestLabels(path, file_name, save_path, factor_dir = 1, factor_files = 1):
    
    number_of_directories = 0
    dir_prefix = "Test"
    
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            if (name[-1] == "t"):
                continue
            number_of_directories += 1

    """for dir_no in range(0,2):
        dir_name = path+"/"+dir_prefix+str(dir_no+1).zfill(3)+"/"
        print(dir_name)
        number_of_files = len(glob.glob1(dir_name,"*.tif"))"""
    
    label_images = [torch.zeros(int(len(glob.glob1(path+"/"+dir_prefix+str(dir_no+1).zfill(3)+"/","*.tif"))/factor_files)) for dir_no in range(int(number_of_directories/factor_dir))]

    with open(path+"/"+file_name) as f:
        lines = f.readlines()   
        lines = [line.rstrip('\n') for line in lines]
        i = 0
        for line in lines:
            if (i==0):
                i +=1
                continue;
            if( i >= int(number_of_directories/factor_dir)):
                break
            splitline = line.split("[")
            if (i==0 or i==1):
                print(splitline)
            splitline = splitline[-1]
            #if (i==0 or i==1):
                #print(splitline)
            splitline = splitline.split("]")[0]
            #if (i==0 or i==1):
                #print(splitline)
            indices = splitline.split(",")
            #if (i==0 or i==1):
                #print(splitline)
            for index_pair in indices:
                index_pair.replace(" ", "")
                index_pair = index_pair.split(":")
                index_1 = int(index_pair[0])-1
                index_2 = int(index_pair[1])-1
                #if(i==3):
                    #print(index_1, index_2)
                label_images[i-1][index_1:index_2+1] = 1
                #if(i==6):
                    #print(label_images[i-1,:])
                
            i += 1
    #print(label_images)
    #print(label_images.shape)
    print("shapes",len(label_images), label_images[0].shape)
    label_images = label_images[:int(number_of_directories/factor_dir)]
    label_images = [image_label[5:] for image_label in label_images]
    print("shapes2",len(label_images), label_images[0].shape)

    #print(label_images.shape)
    #torch.save(label_images[:][:], save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(label_images, f)
    return

def convertUscdToTensor(path, is_train, save_path, factor_dir = 1, factor_files = 1):
    if(is_train):
        dir_prefix = "Train"
        number_of_directories = len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])
    else:
        number_of_directories = 0
        dir_prefix = "Test"
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                if (name[-1] == "t"):
                    continue
                number_of_directories += 1
        

    print(number_of_directories)
    
    dirs = []
    for dir_no in range(0,int(number_of_directories/factor_dir)):
        dir_name = path+"/"+dir_prefix+str(dir_no+1).zfill(3)+"/"
        print(dir_name)
        number_of_files = len(glob.glob1(dir_name,"*.tif"))
        images = []

        for file_no in range(0,int(number_of_files/factor_files)):
            file_name = dir_name+str(file_no+1).zfill(3)+".tif"
            image = Image.open(file_name)
            
            #if((file_no+dir_no == 0)):
                #image.show()   
            new_image = image.resize((224, 224), Image.ANTIALIAS)
            #if((file_no+dir_no == 0)):
                #new_image.show()
            new_image = (ToTensor()(new_image))
            images.append(new_image)
            """if((file_no+dir_no == 0)):
                #print((ToTensor()(image)).shape)
                #print((ToTensor()(image)).permute(1, 2, 0).shape)
                plt.imshow((ToTensor()(image)).reshape(158,238))
                plt.show()"""
        images = torch.stack(images)
        dirs.append(images)
    #for item in dirs:
        #print(item.shape)
    with open(save_path, 'wb') as f:
        pickle.dump(dirs, f)
    #dirs = torch.stack(dirs)
    #torch.save(dirs, save_path)
    return

def tensorToSequence(load_path, save_path):
    #images = torch.load(load_path)#.double()
    #print(images.shape)
    with open(load_path, 'rb') as f:
        dirs = pickle.load(f)

    #[34, 200, 1, 158, 238]
    n_seqs = len(dirs)

    seq_images = list()
    #seq_images = torch.zeros(n_seqs, n_images-5, 3, height, width)
    #seq_images = [torch.zeros(n_images-5, 3, height, width)]

    height = dirs[0].shape[2]
    width = dirs[0].shape[3]
    for image in dirs:
        n_images = image.shape[0]
        
        seq_image = torch.zeros(n_images-5, 3, height, width)
        seq_image[:,0,:,:] = (image[5:,0,:,:] + image[4:-1,0,:,:])/2
        seq_image[:,1,:,:] = (image[3:-2,0,:,:] + image[2:-3,0,:,:])/2
        seq_image[:,2,:,:] = (image[1:-4,0,:,:] + image[0:-5,0,:,:])/2
        seq_images.append(seq_image)
    with open(save_path, 'wb') as f:
        pickle.dump(seq_images, f)
    #torch.save(seq_images, save_path)
    return

def generateTiffImages(video_path, save_path):

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(save_path+"%d.tif" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        if(count%10==0):
            print('Read a new frame: ', success)
        count += 1
    return

def main():
    """
    #convertUscdToTensor(USCD_PATH+USCD1_DIRNAME+"/Train", True, PREPARED_PATH+"uscd1_train.pkl",1,1)
    #tensorToSequence(PREPARED_PATH+"uscd1_train.pkl", PREPARED_PATH+"uscd1seq_train.pkl")

    #convertUscdToTensor(USCD_PATH+USCD2_DIRNAME+"/Train", True, PREPARED_PATH+"uscd2_train.pkl",1,1)
    #tensorToSequence(PREPARED_PATH+"uscd2_train.pkl", PREPARED_PATH+"uscd2seq_train.pkl")

    #convertUscdToTensor(USCD_PATH+USCD1_DIRNAME+"/Train", True, PREPARED_PATH+"sample_uscd1_train.pkl",4,2)
    #tensorToSequence(PREPARED_PATH+"sample_uscd1_train.pkl", PREPARED_PATH+"sample_uscd1seq_train.pkl")

    #convertUscdToTensor(USCD_PATH+USCD1_DIRNAME+"/Test", False, PREPARED_PATH+"uscd1_test.pkl",1,1)
    #tensorToSequence(PREPARED_PATH+"uscd1_test.pkl", PREPARED_PATH+"uscd1seq_test.pkl")

    #convertUscdToTensor(USCD_PATH+USCD2_DIRNAME+"/Test", False, PREPARED_PATH+"uscd2_test.pkl",1,1)
    #tensorToSequence(PREPARED_PATH+"uscd2_test.pkl", PREPARED_PATH+"uscd2seq_test.pkl")

    #convertUscdToTensor(USCD_PATH+USCD1_DIRNAME+"/Test", False, PREPARED_PATH+"sample_uscd1_test.pkl",4,2)
    #tensorToSequence(PREPARED_PATH+"sample_uscd1_test.pkl", PREPARED_PATH+"sample_uscd1seq_test.pkl")

    """
    generateTestLabels(USCD_PATH+USCD1_DIRNAME+"/Test", "UCSDped1.m", PREPARED_PATH+"uscd1seq_labels_test.pkl", 1,1)
    generateTestLabels(USCD_PATH+USCD2_DIRNAME+"/Test", "UCSDped2.m", PREPARED_PATH+"uscd2seq_labels_test.pkl", 1,1)
    generateTestLabels(USCD_PATH+USCD1_DIRNAME+"/Test", "UCSDped1.m", PREPARED_PATH+"sample_uscd1seq_labels_test.pkl", 4,2)

    #generateTiffImages(SUB_PATH+SUBENTRY_DIRNAME+".AVI", SUB_PATH+SUBENTRY_DIRNAME)


if __name__ == '__main__':
    main()