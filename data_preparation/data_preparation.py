import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Tensorflow CUDA warnings will not be shown here 
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("TensorFlow version is:", tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization

print("Modules loaded")
print("----->  Let's go  :)")

# Define ImageDataGenerator
datagen = ImageDataGenerator()

# Define batch size and image size
batch_size = 32
image_size = (150, 150)

# Function to create data generators from DataFrame
def create_datagen(df, directory):
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=directory,
        x_col="image_pathes",
        y_col="labels",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True
    )

# Helper function to create a DataFrame from image paths and labels
def create_dataframe(directory):
    image_pathes = []
    labels = []
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        images = os.listdir(file_path)
        for image in images:
            image_path = os.path.join(file_path, image)
            image_pathes.append(image_path)
            labels.append(file)
    return pd.DataFrame({'image_pathes': image_pathes, 'labels': labels})

# Define paths to datasets
train_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/"
test_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/"
val_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/val/"

# Create DataFrames for training, testing, and validation datasets
train_df = create_dataframe(train_path)
test_df = create_dataframe(test_path)
valid_df = create_dataframe(val_path)

# Create data generators for training, testing, and validation datasets
train_generator = create_datagen(train_df, train_path)
test_generator = create_datagen(test_df, test_path)
valid_generator = create_datagen(valid_df, val_path)
