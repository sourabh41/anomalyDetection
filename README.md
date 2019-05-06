# Deep Anomaly: Anomaly Detection in Crowded Scenes

## Objective

Given a video frame-by-frame, to detect and report the presence and location of anomalous behaviour compared to the normal behaviour in scenes with crowds.

## Solution Approaches

Given frames of normal behaviour in crowded scenes, we determine normal behaviour and train a Gaussian Classifier to detect scenes that lie outside of the normal region.

Features are extracted from a frame using a pretrained alexnet model, and we train two gaussian classifiers. One on the vanilla features, and another one on encoding of the features using a sparse convolutional encoder. The second classifier helps in distinguishing suspicious behaviour which is classified neither abnormal or normal by the first classifier, i.e. which is not a confident-anomaly as categorized by the first classifier.

We tried using various layers of Alexnet and various parameters of the Gaussian Classifier to get the best results. We also varied the number of nodes in the sparse autoencoder and settled at the optimum values for all the 3 in the end.

Out Model captures both location based and temporal relationships.

## Datasets

We used two datasets, UCSD pedestrian dataset and Subway dataset

UCSD : Dominant dynamic objects in this dataset are walkers where crowd density varies from low to high.  An appearing object such as a car, skateboarder, wheelchair, or bicycle is considered to create an anomaly.

Subway : This dataset contains two sequences recorded at the entrance and exit of a subway station.  People entering and exiting the station usually behave normally.  Abnormal events are defined by people moving in the wrong direction, or avoiding payment.


## Model 

![FCN structure for detecting anomalies](/images/model.png?raw=true "Model")

A single frame is represented by the average of its neighbors. The first Gaussian classifier is trained on the last fixed convolutional layer, and the second is trained on the trainable encoder layer.

Note that this approach and model need to be trained differently for different camera settings, hence the most practical use-case of our model lies in security applications (Determining what is normal and what is abnormal given a CCTV footage).

The other benefit of using a Gaussian Model is that we don't have to train on the outliers, which makes a lot of sense for an Anomaly Detection Model since datasets with significant representation of Anomalies are very rare, and using traditional models with such skewed datasets may lead to overtraining on the normal scenes.

Our model uses only the "Normal" Frames for training, and the testing is done on a combination of normal and abnormal scenes, all of which are provided on the dataset.

## Results

More than the accuracy, the best feature of this model is its speed.
Training this model is a one time process and does not require multiple epochs or passes, especially for the first Gaussian Layer, which does quite well alone as well.

More importantly, the testing of this model is very quick. This model is thus capable of pointing out anomalies in real-time cctv footage or industrial-monitoring cameras as soon as something abnormal comes up on the screen.

We obtain a maximum of 67.49% accuracy of detecting abnormal frames in the UCSD dataset using both the Gaussian Classifiers. The first classifier alone gives an accuracy of 51.16%, and thus the presence of the sparse autoencoder greatly enhances the performance of the model.

The testing is almost instantaneous and training is also sufficiently quick and grows linearly with the number of frames used.
Frame by frame example of an anomaly (a person using a skateboard) :

![Frame 1](./images/1.png?raw=true "Frame 1")
![Frame 2](./images/2.png?raw=true "Frame 2")
![Frame 3](./images/3.png?raw=true "Frame 3")
![Frame 4](./images/4.png?raw=true "Frame 4")
![Frame 5](./images/5.png?raw=true "Frame 5")
![Frame 6](./images/6.png?raw=true "Frame 6")
![Frame 7](./images/7.png?raw=true "Frame 7")
![Frame 8](./images/8.png?raw=true "Frame 8")
![Frame 9](./images/9.png?raw=true "Frame 9")


Experimentation with the number of layers led to results coherent with those of the paper.


## Observations

The parameters and the mean feature vectors obtained from the Gaussian Classifier both vary from camera to camera, thus this model requires a separate training for each and every "type" of footage that it works on.

On the other hand, the training of this model does not require much time or many frames to begin with, and thus training individually on each camera is not a very hard task for this model.

We used two pretrained alexnet models for the fixed part of the network, one trained on the imageNet database and other on the places365 database. While we expect the network trained on the places365 database to perform better as it classifies based on the scenes, it performed poorer than the network trained on ImageNet.

An alexnet model trained collectively on the imageNet and the places365 database could be a nice step forward.


## References

* [Deep-Anomaly: Fully Convolutional Neural Network for Fast Anomaly Detection in Crowded Scenes](https://arxiv.org/abs/1609.00866)
