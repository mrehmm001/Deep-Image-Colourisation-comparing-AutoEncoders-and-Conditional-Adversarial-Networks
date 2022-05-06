import os

# DIRECTORY INFORMATION
DATASET = "" # UPDATE (specify the dataset link and then you're set to go!)
TEST_NAME ="test"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/'+DATASET+'/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/'+DATASET+'/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/'+DATASET+'/')
LOG_DIR = os.path.join(ROOT_DIR, 'LOGS/'+DATASET+'/')

TRAIN_DIR = "train/dataset"  # UPDATE
TEST_DIR = "test/dataset" # UPDATE

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 10


# TRAINING INFORMATION
PRETRAINED = "my_model_colorizationEpoch19.h5" # UPDATE
NUM_EPOCHS = 20
