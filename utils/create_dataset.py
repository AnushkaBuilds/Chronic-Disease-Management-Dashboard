# importing the necessary packages
import config
import getPaths
import random
import shutil
import os


# Get the paths to all input images in the original input directory and shuffling them
imagePaths = list(getPaths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# computing the training and testing split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# Further splitting part of the training data for validation
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# Defining the datasets that we'll be building in the final input folder
datasets = [
    ("training", trainPaths, config.TRAIN_PATH),
    ("validation", valPaths, config.VAL_PATH),
    ("testing", testPaths, config.TEST_PATH)
]

# Looping over the datasets
for (dType, imagePaths, baseOutput) in datasets:
    print("Building '{}' split".format(dType))
    if not os.path.exists(baseOutput):
        print("'Creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)
    # Loop over the input image paths
    for inputPath in imagePaths:
        # Extracting the filename of the input image
        # Extracting class label ("0" for "NORMAL" and "1" for "PNEUMONIA")
        filename = inputPath.split(os.path.sep)[-1]
        label = inputPath.split(os.path.sep)[-2]
        # Building the path to the label directory
        labelPath = os.path.sep.join([baseOutput, label])
        if not os.path.exists(labelPath):
            print("'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)
        # Creating the path to the destination image and then copy the image
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)