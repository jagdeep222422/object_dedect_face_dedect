import torch
import cv2
import numpy as np

# Load the YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the image
img = cv2.imread('C:/Users/Pradeep kumar mahato/Desktop/Camera/IMG_20210214_140913.jpg')

# Perform object detection
results = model(img)

# Display the results
results.show()
results