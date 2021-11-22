import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import PIL.Image
import cv2
import numpy as np

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from WheatTestData import test_dataloader

from WheatData import valid_data_loader

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig

def predict():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    model.load_state_dict(torch.load("models_frc/frc.pth"))

    model = model.to(device)
    save_dir = 'predict/'
    
    
    model.eval()
    with torch.no_grad():
        for image_name_batch, test_images in test_dataloader:
            images = list(image.to(device) for image in test_images)
            outputs = model(images)
            for i, image in enumerate(images):
                detection_threshold = 0.5
                sample = image.permute(1,2,0).cpu().numpy()
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()

                boxes = boxes[scores >= detection_threshold].astype(np.int32)


                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                for box in boxes:
                    cv2.rectangle(sample,
                                (box[0], box[1]),
                                (box[2], box[3]),
                                (220, 0, 0), 2)
                ax.set_axis_off()
                ax.imshow(sample)
                plt.savefig(save_dir + image_name_batch[i])
    
if __name__ == "__main__":
    predict()
