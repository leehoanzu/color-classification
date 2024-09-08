import torch

import numpy as np
import onnxruntime

import time
import os

from PIL import Image

import cv2

from torchvision import transforms

class OnnxUtils:
    def __init__(self,  onnx_model_path, labelPath):
        self.labels = self.loadLabels(labelPath)
        self.ortSession = onnxruntime.InferenceSession(onnx_model_path, providers = onnxruntime.get_available_providers()) 
        torch.backends.quantized.engine = 'qnnpack'
    
    def loadLabels(self, label_path):
        # Read list label from .txt file
        # Strip whitespace characters from each label and create a list of stripped labels
        with open(label_path, 'r') as file:
            return [label.strip() for label in file.readlines()]
    
    def preProcessImage(self, inputImage):
        # Pre-procces before putting in network

        # Check type of image 
        # if type of class np.array, we need to change to PIL class
        if isinstance(inputImage, np.ndarray):
            # Convert OpenCV BGR image to RGB (PIL expects RGB)            
            # Convert numpy array to PIL Image
            inputImage = Image.fromarray(cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB))

        preprocess = transforms.Compose([
            transforms.CenterCrop((480, 640)),  # Change image size 480 x 640 (H,W)
            transforms.ToTensor(),   # Adapt the image to tensor representation
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normolize
        ])
        
        return preprocess(inputImage).unsqueeze(0).numpy()
    

    # Perform inference using ONNX Runtime
    # First, we need to put it in onnx session runtime
    # Indicate position of values in array  
    def infer(self, image):
        input_name = self.ortSession.get_inputs()[0].name
        outputs = self.ortSession.run(None, {input_name: image})
        return np.argmax(outputs)
    
    # Show the result
    def display(self, preds):
        print(f'Predicted: {self.labels[preds]}')
    
if __name__ == '__main__':

    def getAbsPath(path):
        # Get absolutely path
        currentDir = os.path.dirname(os.path.abspath(__file__))

        return os.path.abspath(os.path.join(currentDir, path))
    
    # Path of pretrained model
    onnx_model_path = getAbsPath("../model/color-best-edc.onnx")

    # Path of label
    pathLabels = getAbsPath("../label/color-label.txt")

    # Create intance of onnx-runtime
    ortSession = OnnxUtils(onnx_model_path, pathLabels)

    pathImage = getAbsPath("../test-trigger.png")

    image = cv2.imread(pathImage)   
    print(type(image)) 
        # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # # Convert numpy array to PIL Image
        # img_pil = Image.fromarray(img_rgb)


    preImage = ortSession.preProcessImage(image)

    preds = ortSession.infer(preImage)

    ortSession.display(preds)

    for i in range(666, 677, 1):
        # Path of test image
        # pathImage = "/home/edc/Desktop/back-up/test-onnx/image_edc/val/1/white_{i}.png"
        pathImage = getAbsPath(f"../img/val/1/white_{i}.png")

        image = cv2.imread(pathImage)   
        print(type(image)) 

        t0 = time.time()

        preImage = ortSession.preProcessImage(image)

        preds = ortSession.infer(preImage)

        ortSession.display(preds)

        print(f'Done. ({time.time() - t0:.3f}s)')

    print("Build successfully!")