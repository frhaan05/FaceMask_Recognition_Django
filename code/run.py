import os, sys
import cv2 ### pip install opencv-python
import numpy as np ## pip install numpy
import json
from datetime import datetime

# import keras for loading the classifier
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from networks import Facenet
from utils.utils import compare_faces, preprocess_face
from utils.face_alignment import FaceAligner
import CONFIG

assert os.path.exists(CONFIG.FR_FACENET_WEIGHTS), "facenet weights do not exists..."
assert os.path.exists(CONFIG.MOBILENET_MODEL_PATH), "mobilenet weights do not exists..."

assert (os.path.exists(CONFIG.FR_PATH_TO_CLASSES_FILE)
            or os.path.exists(CONFIG.FR_PATH_TO_DATASET_FILE)
            or os.path.exists(CONFIG.FR_PATH_TO_CLASSIFIER)), "required files not found..."
#################################################################################

from mtcnn import MTCNN

if __name__ == "__main__":
    
    # load classes
    with open(CONFIG.FR_PATH_TO_CLASSES_FILE, 'r') as fp:
        classes = json.load(fp)

    # load data
    data = np.load(CONFIG.FR_PATH_TO_DATASET_FILE)
    X = data["features"]
    y = data["labels"]

    print("loading models...")
    # load face detector model
    face_detector = MTCNN()
    # create Face Alignment objects
    face_aligner = FaceAligner(desiredFaceWidth=CONFIG.IMAGE_INPUT_SIZE[0])
    # load facenet model
    facenet = Facenet.loadModel(weights=CONFIG.FR_FACENET_WEIGHTS)
    # load face recognition classifier
    clf = load_model(CONFIG.FR_PATH_TO_CLASSIFIER)
    # load trained classifier
    mask_model = load_model(CONFIG.MOBILENET_MODEL_PATH)
    print("models loaded successfully...")

    print('Opening Webcam...')
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "ERROT -> unable to open the webcam"
    top_n = 3
    while True:
        try:
            # Capture frame-by-frame
            ret, frame_orig = cap.read()
            # resize the frame to increase the speed
            frame = frame_orig.copy()
            # convert the frame into RGB image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # detect faces using MTCNN face detector
            faces = face_detector.detect_faces(frame)

            if len(faces) == 0:
                # print(f"no faces found in {image}")
                continue

            # iterate through every face
            for face in faces:
                # face_image, box, score = face["face"], face["box"], face["score"]
                landmarks, box, score = face["keypoints"], face["box"], face["confidence"]

                # if face confidnece score is less than 60% then skip
                if score < 0.8:
                    continue
                
                # extract the bounding box information from the face
                x1, y1, width, height = box
                x2, y2 = x1 + width, y1 + height

                face_image = frame[y1:y2, x1:x2]
                face_mobilenetv2 = face_image.copy()
                # resize the face to the target size of the classifier
                face_mobilenetv2 = cv2.resize(face_mobilenetv2, CONFIG.IMAGE_INPUT_SIZE)
                # normalize the face array and convert to float 32
                face_mobilenetv2 = (face_mobilenetv2 / 255.).astype(np.float32)
                # add extra dimension to the image as deep learning models accept images in batch
                face_mobilenetv2 = np.expand_dims(face_mobilenetv2, axis=0)
                # perform the predictions
                prediction = mask_model.predict(face_mobilenetv2)[0]
                # get the index if the class with highest probability
                index = np.argmax(prediction)
                # get the label and probability
                label = CONFIG.MOBILENET_CLASSES[index]
                prob = round(prediction[index] * 100, 2)

                # depending on the class, draw the bounding box on the frame
                if label == "Correctly Masked Face":
                    color = (0, 255, 0)
                elif label == "Incorrectly Masked Face":
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                str_ = label + ": " + str(prob) + "%"

                if label in ["Incorrectly Masked Face", "Unmasked Face"]:

                    # first align the face
                    face_facenet = face_aligner.align(frame, landmarks['right_eye'], landmarks['left_eye'])
                  
                    # do face recognition
                    face_facenet = preprocess_face(face_facenet, CONFIG.FACENET_INPUT_SIZE)
                    embeddings = facenet.predict(face_facenet)

                    # predict the person based on embeddings
                    predictions = clf.predict(embeddings)[0]
                    # predictions with top three probs
                    predictions_top_3 = (-predictions).argsort()[:top_n]
                    # taking the top-3 predictions and then extracting their embeddings
                    known_embeddings = np.array([np.mean(X[np.where(y == _idx)], axis=0) for _idx in predictions_top_3])
                    # comparing top-3 predicted person's embeddings with query person
                    result = compare_faces(known_embeddings, embeddings[0].tolist(), tolerance=0.47)
                    # post-processing final result
                    name = classes[str(predictions_top_3[result])] if result is not None else "unknown"
                    str_ = str_ + ' ' + name
                
                # Crop the image frame into rectangle
                cv2.rectangle(frame_orig, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_orig, str_, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

            # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow("predictions", frame_orig)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(str(e))
            
    # release capture
    cap.release()
    cv2.destroyAllWindows()
