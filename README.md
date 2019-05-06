# Anomaly Detection in Crowded Scenes

## Objective

Given a video frame-by-frame, to detect and report the anomalous behaviour compared to normal in crowded areas.

## Solution Approaches

Given frames of normal behaviour in crowded scenes, we determine normal behaviour and train a Gaussian Classifier to detect scenes that lie outside of the normal region.

Features are extracted from a frame using a pretrained alexnet model, and we train two gaussian classifiers. One on the vanilla features, and another one on encoding of the features using a sparse convolutional encoder. The second classifier helps in distinguishing suspicious behaviour which is classified neither abnormal or normal by the first classifier.

## Datasets

We used two datasets, UCSD pedestrian dataset and Subway dataset

UCSD : Dominant dynamic objects in this dataset are walkers where crowd density varies from low to high.  An appearing object such as a car, skateboarder, wheelchair, or bicycle is considered to create an anomaly.

Subway : This dataset contains two sequences recorded at the entrance and exit of a subway station.  People entering and exiting the station usually behave normally.  Abnormal events are defined by people moving in the wrong direction, or avoiding payment.


## Model 

![FCN structure for detecting anomalies](/images/model.png?raw=true "Model")

A single frame is represented by the average of its neighbors. The first Gaussian classifier is trained on the last fixed convolutional layer, and the second is trained on the trainable encoder layer.

## Results

We obtain a maximum of 67.49% accuracy of detecting abnormal frames in the UCSD dataset.
Frame by frame example of an anomaly (a person using a skateboard) :

![Frame 1](./images/008_052.tif?raw=true "Frame 1")
![Frame 2](./images/008_053.tif?raw=true "Frame 2")
![Frame 3](./images/008_054.tif?raw=true "Frame 3")
![Frame 4](./images/008_055.tif?raw=true "Frame 4")
![Frame 5](./images/008_056.tif?raw=true "Frame 5")
![Frame 6](./images/008_057.tif?raw=true "Frame 6")
![Frame 7](./images/008_058.tif?raw=true "Frame 7")
![Frame 8](./images/008_059.tif?raw=true "Frame 8")
![Frame 9](./images/008_060.tif?raw=true "Frame 9")


## Observations

We used two pretrained alexnet models for the fixed part of the network, one trained on the imageNet database and other on the places365 database. While we expect the network trained on the places365 database to perform better as it classifies based on the scenes, it performed poorer than the network trained on ImageNet. 


## References

* [Deep-Anomaly: Fully Convolutional Neural Network for Fast Anomaly Detection in Crowded Scenes](https://arxiv.org/abs/1609.00866)
