# Train AI Model

This guide explains how to train a convolutional neural network (CNN) using transfer learning for image classification. The goal is to classify images into predefined color categories such as white, black, blue, etc.

<!-- We follow [`PyTorch tutorial`](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#creating-models) to create our AI model by dividing the 3 color range into 3 distinct folders. Each folder represents a different color, providing a structured way to organize the data.  -->

## Datasets

* We create our AI model by dividing the 3 color range into 3 distinct folders. Each folder represents a different color, providing a structured way to organize the data. 

```
dataset/
├── 0/
│   ├── blue_00.jpg
│   ├── blue_01.jpg
│   └── ...
├── 1/
│   ├── white_00.jpg
│   ├── white_01.jpg
│   └── ...
└── 2/
    ├── black_00.jpg
    ├── black_01.jpg
    └── ...
```

## Loading the Datasets

* Use torchvision.datasets.ImageFolder to load the images, and torchvision.transforms to resize and normalize them:

```python
# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "/content/drive/MyDrive/STUDY/DeepStream/image"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=2)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes # (0: blue, 1: white, 2: black)

print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Train the Model

### General function

* we write a general function to train a model:

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Specify the path to save the best model parameters
    best_model_params_path = '/content/drive/MyDrive/STUDY/DeepStream/color-best-edc.pt'
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)#.float()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() 
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model
```

### Finetuning the ConvNet

* Load a pretrained model and reset final fully connected layer:

```python
# Load pretrained model
model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 3)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as pposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

### Train and Evaluate

* It will take around 30 minutes to finish.

```python
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

<p align="center">
  <img src="https://github.com/leehoanzu/color-classification/blob/main/screen-shots/prediction-result.png" alt="prediction-result" width="45%" height="auto">
  <img src="https://github.com/leehoanzu/color-classification/blob/main/screen-shots/heatmap.png" alt="heatmap" width="45%" height="auto">
</p>

## Inference on custome images

* Use the trained model to make predictions on custom images and visualize the predicted class labels along with the images.

```python
# Create an instance of ResNet18
model_conv = models.resnet18()
# Replace the fully connected layer with a new one having 2 output units
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 3)
model_conv = model_conv.to(device)
# Load the state dictionary into the ResNet18 model
state_dict = torch.load('/content/drive/MyDrive/STUDY/train-model/model/color-best.pt')

# Load the adapted state dictionary into the model
model_conv.load_state_dict(state_dict)

model_conv.eval()  # Set the model to evaluation mode
visualize_model_predictions(
    model_conv,
    img_path='/content/drive/MyDrive/STUDY/train-model/img/val/1/white_0.png'
)

plt.ioff()
plt.show()
```

![prediction-result](https://github.com/leehoanzu/color-classification/blob/main/screen-shots/inference-results.png)

