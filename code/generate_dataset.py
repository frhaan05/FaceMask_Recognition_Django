import os
import cv2 #pip install opencv-python
import numpy as np
from tqdm import tqdm #pip install tqdm

from networks import RatinaFaceWrapper
import CONFIG

# set paths
DATASET_PATH = "dataset/Custom_raw"
DISTINATION_PATH = "dataset/Custom"

# check if the path exists, if not the throw error
assert os.path.exists(DATASET_PATH), "ERROR -> Dataset Path doesn't exists"

# make directory for output images
if not os.path.exists(DISTINATION_PATH):
    os.makedirs(DISTINATION_PATH)

# build the MTCNN model
print("building model...")
detector = RatinaFaceWrapper.build_model()
print("model built successfully...")

def gen_raw_dataset():
    counter = 0
    # read images one by one in the Dataset Path
    for file in tqdm(os.listdir(DATASET_PATH)):
        # file_path is the complete path to an image
        file_path = os.path.join(DATASET_PATH, file)
        
        # The directory is found only for Incorrectly Masked Face as this class contains sub classes. Therefore, we need to 
        # iterate again on each sub class
        # if directory is found then go deep inside
        if not os.path.isdir(file_path):
            continue

        # make directory for sub distination path
        if not os.path.exists(os.path.join(DISTINATION_PATH, file)):
            os.makedirs(os.path.join(DISTINATION_PATH, file))
        
        # iterate through images in sub class
        for sub_file in os.listdir(file_path):
            try:
                basename, extension = os.path.splitext(sub_file)
                if extension not in CONFIG.IMAGE_FORMATS:
                    continue
                
                image_path  = os.path.join(file_path, sub_file)

                # read the image. Convert into numpy array
                img = cv2.imread(image_path)

                # detect a face in the image
                faces = RatinaFaceWrapper.extract_faces(detector, img, align=True)
                # faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                for face in faces:
                    x, y, w, h = face['box']

                    # check if face is not too small
                    if w < 60 or h < 60:
                        continue
                
                    # extract the top-left corner and wight and height of the bounding box.
                    # MTCNN returns face information in this format
                    x, y, w, h = face['box']
                    
                    # crop the face from the image
                    face = img[y:y + h, x:x + w]
                    
                    # resizing the face to target_size
                    face = cv2.resize(face, CONFIG.IMAGE_INPUT_SIZE)
                    
                    # save the face in the distination path
                    dst_image_path = os.path.join(DISTINATION_PATH, file, sub_file)
                    cv2.imwrite(dst_image_path, face)
                    counter += 1
            except Exception as e:
                pass
            
    print(f"number of images generated: {counter}")
    print(f"resized dataset has been generated in: {DISTINATION_PATH}")


def gen_MaskedFaceNET_dataset():
    # build the MTCNN model
    print("building model...")
    detector = MTCNN()
    print("model built successfully...")

    counter = 0
    # read images one by one in the Dataset Path
    for file in tqdm(os.listdir(DATASET_PATH)):
        
        # file_path is the complete path to an image
        file_path = os.path.join(DATASET_PATH, file)

        try:
            image_path  = file_path

            # read the image. Convert into numpy array
            img = cv2.imread(image_path)
                    
            # detect a face in the image
            faces = RatinaFaceWrapper.extract_faces(detector, img, align=True)

            if len(faces) == 0:
                continue

            # face_area is area of a face with largest area
            face_area = max([(face['box'][2] * face['box'][3]) for face in faces])
            face_final = [face for face in faces if (face['box'][2] * face['box'][3]) == face_area][0]
            
            # extract the top-left corner and wight and height of the bounding box.
            # MTCNN returns face information in this format
            x, y, w, h = face_final['box']
            
            # crop the face from the image
            face = img[y:y + h, x:x + w]
            
            # resizing the face to target_size
            face = cv2.resize(face, CONFIG.IMAGE_INPUT_SIZE)
            
            # save the face in the distination path
            dst_image_path = os.path.join(DISTINATION_PATH, file)
            cv2.imwrite(dst_image_path, face)
            counter += 1
            
        except Exception as e:
            pass
            
    print(f"number of images generated: {counter}")
    print(f"resized dataset has been generated in: {DISTINATION_PATH}")


if __name__ == "__main__":
    # generate model compatible dataset
    gen_raw_dataset()

    # generate model compatible dataset from maskedFaceNet
    # gen_MaskedFaceNET_dataset()