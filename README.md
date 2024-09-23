# Pedestrian Detection with DINO
This repository contains code and resources for pedestrian detection using the DINO model. The model is trained for object detection and tested on a custom pedestrian dataset. This README walks through setup, running inference, visualization of the results, and troubleshooting.

# Table of Contents
    Requirements
    Installation
    Model Setup
    Dataset Preparation
    Running Inference
    Visualization
    Troubleshooting
    References
# Requirements
Ensure you have the following libraries installed in your environment:

    torch
    torchvision
    numpy
    matplotlib
    PIL
    pycocotools
# You can install these dependencies by running:

    pip install torch torchvision numpy matplotlib pillow pycocotools
Make sure to use a machine with CUDA-enabled GPU to take advantage of faster inference with GPU acceleration.

# Installation
## Clone this repository:


    git clone https://github.com/yourusername/pedestrian-detection-dino.git
    cd pedestrian-detection-dino
Install required dependencies:


    pip install -r requirements.txt
Set up the DINO model and its dependencies by running:

    python setup.py build_ext --inplace
Download the pre-trained weights or checkpoints for the model (ensure the checkpoint path is properly configured in the code).

# Model Setup
To run inference on the model, follow these steps to load the trained weights and initialize the DINO model.

## Loading the Model

    import torch
    from models import build_model  # Adjust based on your structure

## Path to the trained model checkpoint
    checkpoint_path = "/path/to/your/checkpoint.pth"

## Load the DINO model and its components
    model, criterion, postprocessors = build_model(args)
    model = model.cuda()
    model.eval()

## Load the trained weights into the model
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
## Dataset Preparation
Before running the inference, make sure your dataset (images and annotations) is correctly structured. Update paths in the code to point to your pedestrian dataset:


### In datasets/coco.py
    def build(image_set, args):
        root = Path(args.coco_path)
        mode = 'instances'
        PATHS = {
            "train": (root / "train_images", root / "annotations" / 'train_annotations.json'),
            "val": (root / "val_images", root / "annotations" / 'val_annotations.json'),
        }
Ensure you adjust the file paths in the code to point to your custom annotations.

# Running Inference
The following script demonstrates how to run inference on a test image:

## Preprocessing and Inference

    from PIL import Image
    from torchvision import transforms

## Define preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

## Preprocess the input image
    def preprocess_image(image_path):
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)  # Add batch dimension

## Run inference on the preprocessed image
    @torch.no_grad()
    def run_inference(model, image_tensor):
        image_tensor = image_tensor.cuda()
        outputs = model(image_tensor)
        return postprocessors['bbox'](outputs, torch.Tensor([[1.0, 1.0]]).cuda())
# Visualization
After running inference, you can visualize the bounding boxes and scores on the test images:

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

## Visualize the predicted bounding boxes
    def visualize_prediction(image_path, outputs):
        image = Image.open(image_path)
        pred_boxes = outputs['boxes'].cpu().numpy()
        pred_scores = outputs['scores'].cpu().numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        plt.show()

## Example usage

    image_path = '/path/to/your/test_image.jpg'
    image_tensor = preprocess_image(image_path)
    outputs = run_inference(model, image_tensor)
    visualize_prediction(image_path, outputs)
Troubleshooting
### 1. FileNotFoundError: [Errno 2] No such file or directory
#### Cause: 
This occurs when the annotation file or image path is incorrect.
#### Solution: 
Ensure that the dataset paths in the build_dataset function point to the correct directories and files.
Example:


    "train": (root / "train_images", root / "annotations" / 'train_annotations.json')
Update the paths to match your dataset structure.

### 2. CUDA Error: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
#### Cause: 
The GPU driver or CUDA environment is not properly set up.
#### Solution: 
Ensure that your runtime environment in Google Colab or locally is set to use GPU. 
In Colab, go to Runtime > Change runtime type and select GPU.
### 3. RuntimeError: Size Mismatch in Model Layers
#### Cause: 
The model's output layer does not match the number of classes in your dataset.
#### Solution: 
Update the num_classes parameter when building the model, ensuring it matches the number of classes in your dataset (including background).
#### Example:

    args.num_classes = 2  # For pedestrian (class 1) and background (class 0)
### 4. Model Loading Error: NotImplementedError: 'Cuda is not available'
#### Cause: 
CUDA is not properly installed or available in the runtime environment.
#### Solution: 
Make sure CUDA is available by running:

    import torch
    print(torch.cuda.is_available())
If it returns False, ensure the runtime is configured with a GPU.
# References

#### DINO: DETR with Improved Deformable Attention
#### COCO Dataset Format
#### PyTorch Documentation
# License
This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.


## Key Notes for the `README.md`:
- **Installation**: Simple steps for setting up the environment.
- **Model Setup and Dataset Preparation**: Guides to ensure paths and models are correctly set up.
- **Inference and Visualization**: Clear steps on running predictions and visualizing results.
- **Troubleshooting**: Covers common errors such as file paths, CUDA issues, and size mismatches with actionable solutions.

This `README.md` can serve as a clear guide for anyone cloning your repository and getting started with pedestrian detection using the DINO model.
