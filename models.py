import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def get_labels():
    return [
        'Apple scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf',
        'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf', 'Corn gray leaf spot',
        'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf early blight',
        'Potato leaf late blight', 'Raspberry leaf', 'Soyabean leaf',
        'Squash powdery mildew leaf', 'Strawberry leaf', 'Tomato early blight leaf',
        'Tomato septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot',
        'Tomato leaf late blight', 'Tomato leaf mosaic virus',
        'Tomato leaf yellow virus', 'Tomato mold leaf', 'Grape leaf', 'Grape leaf black rot'
    ]

def get_transforms(image_size):
    inference_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return inference_transform

def denormalize(
    x,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)

def getPlantDiseaseName(output_class):
    class_name = get_labels()[int(output_class)]
    plant_name = class_name.split(' ')[0]
    disease_name = ' '.join(class_name.split(' ')[1:])

    return [plant_name, disease_name]

@st.cache_resource
def get_model():
    model = models.resnet50()

    for params in model.parameters():
        params.requires_grad = False

    num_in_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_in_ftrs, 27)

    weights = torch.load('best_model.pth', map_location=torch.device('cpu'))
    
    if weights is not None:
        model.load_state_dict(weights['model_state_dict'])
    
    return model

def inference(model, image):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.eval()
    
    with torch.no_grad():
        image = image.to(DEVICE)

        # Forward pass.
        outputs = model(image)

    # Softmax probabilities.
    predictions = F.softmax(outputs, dim=1).cpu().numpy()

    # Predicted class number.
    output_class = np.argmax(predictions)
    
    plant_name, disease_name = getPlantDiseaseName(output_class)

    return [plant_name, disease_name, np.max(predictions)]

def inference2(model, image, n_highest_pred):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.eval()
    
    with torch.no_grad():
        image = image.to(DEVICE)

        # Forward pass.
        outputs = model(image)

    # Softmax probabilities.
    predictions = F.softmax(outputs, dim=1).cpu().numpy()
    
    labeled_and_sorted = get_disease_sorted(get_labels(), predictions[0], n_highest_pred)

    return labeled_and_sorted

def get_disease_sorted(labels, predictions, n_highest_pred=0):
    join_list = list(zip(labels, predictions))
    join_list.sort(key=lambda x:x[1], reverse = True)

    if n_highest_pred>0:
        result = join_list[:n_highest_pred]
    else:
        result = join_list

    return result

def predict(image_path):
    model = get_model()

    image = Image.open(image_path)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transforms = get_transforms(224)
    image = transforms(image)
    image = torch.unsqueeze(image, 0)

    return inference2(model, image, 3)