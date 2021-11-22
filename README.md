# wheat-detection-network
wheat detection network

# about the app
We have upload an application based on this network to Apple App Store. You can search it by "Wheat Detector".
Before it passed Apple’s review, you can see a screenshot of this app in the project root directory.

# Dataset
Global Wheat Detection on Kaggle
https://www.kaggle.com/c/global-wheat-detection

# About the code
1. fpn folder contains two .py files: train.py and predict.py
2. WheatData.py contains the code for processing the data set, including to tensor, simple data augmentation and generating data loader.
3. wdn.py contains all the innovations mentioned in the paper, including oof, mixup, test time augmentation, optimization for NMS.
