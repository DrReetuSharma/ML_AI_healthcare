from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Define ImageDataGenerator
datagen = ImageDataGenerator()

# Define batch size and image size
batch_size = 32
image_size = (150, 150)

# Define paths to datasets
train_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/"
test_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/"
val_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/val/"

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

# Load DataFrame for training, testing, and validation datasets
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")
valid_df = pd.read_csv("valid_data.csv")

# Create data generators for training, testing, and validation datasets
train_generator = create_datagen(train_df, train_path)
test_generator = create_datagen(test_df, test_path)
valid_generator = create_datagen(valid_df, val_path)
