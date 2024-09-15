# ONNX Export User Guide

This guide explains how to convert a model defined in PyTorch into the ONNX format using the TorchScript torch.onnx.export ONNX exporter. 
The exported model will be executed with ONNX Runtime. ONNX Runtime is a performance-focused engine for ONNX models, which inferences efficiently across multiple platforms and hardware (Windows, Linux, and Mac and on both CPUs and GPUs).

The following Python script demonstrates how to load a pre-trained model, preprocess an image, export the model to ONNX format, and perform inference using the ONNX model.

## Prerequisites

For this tutorial, you will need to install ONNX and ONNX Runtime. You can get binary builds of ONNX and ONNX Runtime with:

```bash
$ pip install onnx onnxruntime
```

## Load Pre-Trained Model

The loadModel function loads a pre-trained ResNet-18 model and modifies the last fully connected layer to have 3 output features (suitable for a 3-class classification problem). The model is then loaded with pre-trained weights from a local path and set to evaluation mode.

```python
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
```

## Export Model to ONNX

The exportOnnx function converts the PyTorch model to ONNX format. It uses a dummy input image to define the input shape and performs the export using torch.onnx.export().

```python
def exportOnnx(model):
    batch_size = 1
    dummy_input = preProcessImage(getAbsPath("../img/white/white_702.png")) # get the real sample
    dummy_input.to(DEVICE)

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
```

> [!NOTE]  
> <sup>- Input/Output names: You can specify names for the model inputs and outputs (`input_names` and `output_names`)</sup><br>
> <sup>- Dynamic axes: This allows flexible batch sizes during inference by specifying the dynamic axes for the input and output.</sup>

## Perform Preprocessing

Before feeding an image to the model, it should be preprocessed. The `preProcessImage` function resizes, normalizes, and converts an image to a tensor.

```python
def preProcessImage(pathImage):
    inputImage = Image.open(pathImage)

    # Before proceeding with any image processing tasks,
    # it's crucial to prepare or pre-process the image appropriately

    preprocess = transforms.Compose([
        transforms.CenterCrop((480, 640)),  # Change image size (H,W)
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),   # Adapt the image to tensor representation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    return preprocess(inputImage)
```

## Run Inference with ONNX Runtime

After converting the model, you can run inference using ONNX Runtime. The `loadOnnxModel` function loads the ONNX model, and the `inferrence` function performs inference.

```python
def loadOnnxModel(PathOnnxModel):
    return onnxruntime.InferenceSession(PathOnnxModel, providers=onnxruntime.get_available_providers())

def inferrence(ortSession, imageY):
    imageY = imageY.unsqueeze(0)
    ortInput = {ortSession.get_inputs()[0].name: imageY.numpy()}
    ortOutputs = ortSession.run(None, ortInput)
    return ortOutputs
```

## Run the Script

You can execute [the script](https://github.com/leehoanzu/color-classification/blob/main/train/onnxExport.py), simply run:

```bash
$ python3 onnxExport.py
```

## Conclusion

This guide shows how to convert a PyTorch model to ONNX, perform preprocessing, and run inference using ONNX Runtime. This can be particularly useful for deploying models on different platforms.
