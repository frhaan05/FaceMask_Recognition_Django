from scipy.spatial.distance import cosine
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization


def compare_two_faces(known_embedding, candidate_embedding, tolerance=0.4):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return True if cosine(known_embedding, candidate_embedding) <= tolerance else False


def compare_faces(known_embeddings, candidate_embedding, tolerance=0.4):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    # calculate distance between embeddings
    distances = [cosine(known_embedding, candidate_embedding) for known_embedding in known_embeddings]
    min_dist = min(distances)
    min_dist_index = distances.index(min_dist)
    return min_dist_index if min_dist <= tolerance else None

def preprocess_face(face, target_size=(160, 160)):
    # TODO: resize causes transformation on base image, you should add black pixels to rezie it to target_size 
    image = Image.fromarray(face)
    image = image.resize(target_size)
    face_array = np.asarray(image).astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean) / std
    # transform face into one sample
    face_array = np.expand_dims(face_array, axis=0)
    return face_array


def Classifier(input_dim, outputs):
    clmodel = Sequential()
    clmodel.add(Dense(units=200, input_dim=input_dim, kernel_initializer='glorot_uniform'))
    clmodel.add(BatchNormalization())
    clmodel.add(Activation('tanh'))
    clmodel.add(Dropout(0.5))
    clmodel.add(Dense(units=100, kernel_initializer='glorot_uniform'))
    clmodel.add(BatchNormalization())
    clmodel.add(Activation('tanh'))
    clmodel.add(Dropout(0.4))
    clmodel.add(Dense(units=10, kernel_initializer='glorot_uniform'))
    clmodel.add(BatchNormalization())
    clmodel.add(Activation('tanh'))
    clmodel.add(Dropout(0.2))
    clmodel.add(Dense(units=outputs, kernel_initializer='he_uniform'))
    clmodel.add(Activation('softmax'))

    clmodel.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        optimizer='nadam', 
        metrics=['accuracy']
        )
    return clmodel