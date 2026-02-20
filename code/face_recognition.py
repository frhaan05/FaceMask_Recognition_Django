import numpy as np
import os
import json
import shutil
import cv2 
from mtcnn import MTCNN

from networks import Facenet
from utils.utils import Classifier, preprocess_face
from utils.face_alignment import FaceAligner
import CONFIG


assert os.path.exists(CONFIG.FR_FACENET_WEIGHTS), "facenet weights do not exists..."
face_detector = MTCNN()
# create Face Alignment objects
face_aligner = FaceAligner(desiredFaceWidth=CONFIG.IMAGE_INPUT_SIZE[0])
facenet = Facenet.loadModel(weights=CONFIG.FR_FACENET_WEIGHTS)
#################################################################################


def add_to_dataset(person, images_list):
    classes = dict()
    if os.path.exists(CONFIG.FR_PATH_TO_CLASSES_FILE):
        # load classes
        with open(CONFIG.FR_PATH_TO_CLASSES_FILE, 'r') as fp:
            classes = json.load(fp)
        if person in classes.values():
            print("person already exists...")
            return None, False

    user_embeddings = []
    # extract embeddings
    os.makedirs(os.path.join(CONFIG.FR_PATH_TO_MEDIA, person), exist_ok=True)
    for i, image_path in enumerate(images_list):
        if not os.path.exists(image_path):
            continue
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # detect faces using MTCNN face detector
            faces = face_detector.detect_faces(img)
            # face_area is area of a face with largest area
            face_area = max([face['box'][2] * face['box'][3] for face in faces])
            face_final = [face for face in faces if face['box'][2] * face['box'][3] == face_area][0]
            xmin, ymin, w, h = face_final["box"]
            xmax, ymax = xmin + w, ymin + h
            face_image = img[ymin:ymax, xmin:xmax]
            face_image = preprocess_face(face_image, CONFIG.FACENET_INPUT_SIZE)
            embeddings = facenet.predict(face_image)[0].tolist()
            user_embeddings.append(embeddings)
            # save crop
            cv2.imwrite(os.path.join(CONFIG.FR_PATH_TO_MEDIA, person, f'{i}__image.jpg'), 
                cv2.resize(img[ymin:ymax, xmin:xmax], CONFIG.FACENET_INPUT_SIZE))
        except Exception as e:
            print(str(e))

    if os.path.exists(CONFIG.FR_PATH_TO_DATASET_FILE):
        data = np.load(CONFIG.FR_PATH_TO_DATASET_FILE)
        X, y = data["features"], data["labels"]
        print(f"x shape: {X.shape}, classes: {len(np.unique(y))}")
        
        if len(X) > 0:
            X = np.append(X, user_embeddings, axis=0)
            class_to_add = np.max(np.unique(y)) + 1
            labels_to_add = np.array([class_to_add for _ in range(len(user_embeddings))])
            y = np.append(y, labels_to_add, axis=0)
            classes[str(class_to_add)] = person
        else:
            X = np.array(user_embeddings)
            y = np.array([0 for _ in range(len(user_embeddings))])
            classes[str(0)] = person
    else:
        X = np.array(user_embeddings)
        y = np.array([0 for _ in range(len(user_embeddings))])
        classes[str(0)] = person
        
    np.savez(CONFIG.FR_PATH_TO_DATASET_FILE, features=X, labels=y)
    with open(CONFIG.FR_PATH_TO_CLASSES_FILE, "w") as fp:
        json.dump(classes, fp)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("training classifier...")
    train_classifier(X, y)
    return classes, True

def delete_from_dataset(person_to_delete):
    if not os.path.exists(CONFIG.FR_PATH_TO_DATASET_FILE) or not os.path.exists(CONFIG.FR_PATH_TO_CLASSES_FILE):
        return None, False
    # load classes
    with open(CONFIG.FR_PATH_TO_CLASSES_FILE, 'r') as fp:
        classes = json.load(fp)

    data = np.load(CONFIG.FR_PATH_TO_DATASET_FILE)
    X, y = data["features"], data["labels"]
    print(f"x shape: {X.shape}, classes: {len(np.unique(y))}")

    if person_to_delete in classes.values():
        
        idx = int(list(classes.keys())[list(classes.values()).index(person_to_delete)])
        print("person in dataset exists...")
        
        indices = np.where(y == idx)
        X = np.delete(X, indices, axis=0)
        y = np.delete(y, indices, axis=0)
        print("after deleting:", X.shape, len(y))
        np.savez(CONFIG.FR_PATH_TO_DATASET_FILE, features=np.array(X), labels=np.array(y))

        del classes[str(idx)]
        with open(CONFIG.FR_PATH_TO_CLASSES_FILE, "w") as fp:
            json.dump(classes, fp)
        
        if len(classes) <= 1:
            # os.remove(CONFIG.FR_PATH_TO_CLASSES_FILE)
            os.remove(CONFIG.FR_PATH_TO_CLASSIFIER)
            # os.remove(CONFIG.FR_PATH_TO_DATASET_FILE)
        else:
            # train classifier on remaining data
            train_classifier(X, y)
        # delete person's folder from dataset folder
        dir_to_del = os.path.join(CONFIG.FR_PATH_TO_MEDIA, person_to_delete)
        shutil.rmtree(dir_to_del)
        return classes, True
    else:
        print("person doesn't exists...")
        return None, False

def train_classifier(X, y):
    number_classes = len(np.unique(y))
    clr = Classifier(X.shape[1], number_classes)
    clr.fit(
        X,
        y,
        epochs=100,
        verbose=2
    )
    # save the model
    clr.save(CONFIG.FR_PATH_TO_CLASSIFIER)
