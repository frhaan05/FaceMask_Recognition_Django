""" General Configurations """
# define the format of images to be supported
IMAGE_FORMATS = ["jpg", "jpeg", "JPG"]
IMAGE_INPUT_SIZE = (224, 224)

# specify paths
CHECKPOINTS_PATH = "./checkpoints"
LOG_DIR = f"{CHECKPOINTS_PATH}/logs"

""" MOBILENETv2 parameters """

FACE_MASK_DATASET_PATH = "dataset/Face_Mask_Classification"
# Model paths
MOBILENET_MODEL_PATH = f"{CHECKPOINTS_PATH}/mobileNetv2_30_epochs.h5"

# specify classes
MOBILENET_CLASSES = [
           "Correctly Masked Face",
           "Incorrectly Masked Face",
           "Unmasked Face"
]

# training parameters
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 16
EPOCHS = 30


""" Face Recognition Configurations """

# Set Paths to files
FR_PATH_TO_MEDIA = "db/FR/media"
FR_PATH_TO_DATASET_FILE = "db/FR/dataset.npz"
FR_PATH_TO_CLASSES_FILE = "db/FR/people.json"
FR_PATH_TO_CLASSIFIER = "checkpoints/FR_classifier.h5"
FR_FACENET_WEIGHTS = "checkpoints/facenet.h5"

FACENET_INPUT_SIZE = (160, 160)