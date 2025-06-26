# import the packages
import os

# Mounting the google drive
# from google.colab import drive
# drive.mount('/content/gdrive')

# Google Drive Path
# gDrivePath = "gdrive/MyDrive/Colab Notebooks/chest-xray-pneumonia/"

# Project path
PROJ_PATH = "/content"

# Original input directory of images
ORIG_INPUT_DATASET = "/content/dataset/oid"

# Final Derived input directory of images
FINAL_INPUT_DATASET = "/content/dataset/fid"

# Output Path Directory
outputPath = "/content/output"

# Defining the directories
TRAIN_PATH = os.path.sep.join([FINAL_INPUT_DATASET, "training"])
VAL_PATH = os.path.sep.join([FINAL_INPUT_DATASET, "validation"])
TEST_PATH = os.path.sep.join([FINAL_INPUT_DATASET, "testing"])

# Setting the train split to 80%
TRAIN_SPLIT = 0.8

# Setting the val split to 10%
VAL_SPLIT = 0.1

CLASSES = ["normal","pneumonia"]

# Setting the parameters
BATCH_SIZE = 16
INIT_LR = 1e-4
EPOCHS = 20

# Setting the path to save the serialized model after training
MODEL_PATH = os.path.sep.join([outputPath, "PneumoniaPrediction.model"])

# Setting the path for the output training history plots
PLOT_PATH = os.path.sep.join([outputPath, "TrainingHistoryPlot.png"])
