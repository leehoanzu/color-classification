### EXPORT MODEL PYTORCH TO ONNX FORTMAT ###

"""
This is util to use export PyTorch to Onnx

- Step 1: We load PyTorch model on local disk
- Step 2: Preproces image input 
- Step 3: Create an intance onnx-runtime
- Step 4: Input image and start testing
"""

from PIL import Image 

from torchvision import models, transforms
import torch.nn as nn
import torch

import numpy as np
import onnxruntime
import onnx

import time

import os

def loadModel(preTrainnedModelPath):
    # Create device
    print(DEVICE)

    # Create instance of model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features

    # Change the last number of feauters 3
    model.fc = nn.Linear(num_ftrs, 3)
    
    # Load your pre trainned model from device
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(preTrainnedModelPath, map_location=torch.device(DEVICE)))

    # Set the model to evaluation mode
    model.eval()

    return model

def exportOnnx(model):
    batch_size = 1
    dummy_input = preProcessImage(getAbsPath("../img/white/white_702.png")) # get the real sample
    dummy_input.to(DEVICE)
    # sampleX = torch.randn(batch_size, 3, 247, 730, requires_grad=True) # add new dimesion at zero position
    sampleX = dummy_input.unsqueeze(0)

    # opset_version = 11
    torch.onnx.export(model,                                        # model being run
                  sampleX,                                          # model input (or a tuple for multiple inputs)
                  onnx_model_path,                                  # where to save the model (can be a file or file-like object)
                  export_params=True,                               # store the trained parameter weights inside the model file
                  opset_version=10,                                 # the ONNX version to export the model to
                  do_constant_folding=True,                         # whether to execute constant folding for optimization
                  input_names = ['input'],                          # the model's input names
                  output_names = ['output'],                        # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                'output' : {0 : 'batch_size'}})

    return onnx.load(onnx_model_path)

def loadOnnxModel(PathOnnxModel):
    # Load the ONNX model using ONNX Runtime
    return  onnxruntime.InferenceSession(PathOnnxModel, providers = onnxruntime.get_available_providers()) # ["CPUExecutionProvider"]

def preProcessImage(pathImage):
    inputImage = Image.open(pathImage)
    # print(np.array(Image.open(pathImage)).shape)

    # Before proceeding with any image processing tasks,
    # it's crucial to prepare or pre-process the image appropriately

    preprocess = transforms.Compose([
        transforms.CenterCrop((480, 640)),  # Change image size (H,W)
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),   # Adapt the image to tensor representation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    
    return preprocess(inputImage)

def inferrence(ortSession, imageY):
    imageY = imageY.unsqueeze(0)

    # Perform inference using ONNX Runtime
    ortInput = {ortSession.get_inputs()[0].name: imageY.numpy()}
    ortOutputs = ortSession.run(None, ortInput)
    
    return ortOutputs

def loadLabels(labelPath):
    # Read list label from .txt file
    with open(labelPath, 'r') as file:
        # Read all lines from the file into a list
        labels = file.readlines()
        
        # Strip whitespace characters from each label and create a list of stripped labels
        labels = [label.strip() for label in labels]

    # Return the list of stripped labels
    return labels

def getAbsPath(path):
    # Get absolutely path
    currentDir = os.path.dirname(os.path.abspath(__file__))

    return os.path.abspath(os.path.join(currentDir, path))

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    torch.backends.quantized.engine = 'qnnpack'

    # Check device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Path of pretrained model
    preTrainnedModelPath = getAbsPath("../model/color-best-edc.pt")

    # Path of onnx model
    onnx_model_path = getAbsPath("../model/color-best-edc.onnx")

    # Path of label
    pathLabels = getAbsPath("../label/color-label.txt")

    # EXPORT MODEL TO ONNX
    # Load pre trainned model in your local disk
    model = loadModel(preTrainnedModelPath)

    # Export torch model to type of onnx model 
    onnxModel = exportOnnx(model)
    # Parse onnx model, if it errors, the note will appear
    onnx.checker.check_model(onnxModel)

    ortSession = loadOnnxModel(onnx_model_path)
    # print(ortSession)

    for i in range(666, 677, 1):
        print(f"{i}\n")
        # Start unferencing
        t0 = time.time()

        pathImage =  getAbsPath(f"../img/val/1/white_{i}.png") #/home/edc/Desktop/onnx/picture/image/Image_0000.jpg
        imageY = preProcessImage(pathImage)

        # Inferencing your image
        outputY = inferrence(ortSession, imageY)
        # print(outputY)

        # Change list object to numpy object
        out = np.array(outputY)
        # Indicate position of values in array
        preds = np.argmax(out)
        # print(preds)

        # Load label to display terminal monitor
        label = loadLabels(pathLabels)
        print(f'Predicted: {label[preds]}')
        
        print(f'Done. ({time.time() - t0:.3f}s)')

    # Annouccing success
    print("Build succesfully!")

