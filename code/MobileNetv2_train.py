import os, glob
import cv2 ### pip install opencv-python
import numpy as np ## pip install numpy
from tqdm import tqdm

## Import packages for Deep LEARNING MODEL BUIDING AND TRAINING - TRANSFER LEARNING
import tensorflow as tf ## pip install tensorflow OR pip install tensorflow-gpu
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping

# Import packages for plotting dataset samples
import matplotlib.pyplot as plt  ## pip install matlplotlib
import random

# import sklearn packages
from sklearn.model_selection import train_test_split
import CONFIG

def load_dataset():
    X, y = [], []
    stats = {
        "total_images": 0,
        "training_images": 0,
        "testing_images": 0,
        "image_size": None,
        "Correctly Masked Face": 0,
        "Incorrectly Masked Face": 0,
        "Unmasked Face": 0
    }
    # iterate through images one by one
    for image in tqdm(glob.glob(os.path.join(CONFIG.FACE_MASK_DATASET_PATH, "**"), recursive=True)):
        
        # if extension is not in the formats defines, then skip the image
        basename, extension = os.path.splitext(image)
        if extension[1:] not in CONFIG.IMAGE_FORMATS:
            continue

        # splitting the image path into parts and then extracting the label
        path = os.path.normpath(image)
        parts = path.split(os.sep)
        # extract label from folder structure
        label = parts[-2]
        # then read the image, convert into RGB image
        # since the images are already in 224x224 dimension, we don't need to resize them again
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # append the image and label to X and y respectively
        X.append(img)
        y.append(CONFIG.MOBILENET_CLASSES.index(label))
        stats[label] += 1
        stats["total_images"] += 1

    # convert the lists into arrays
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG.VALIDATION_SPLIT, random_state=42)
    stats["training_images"] = X_train.shape[0]
    stats["testing_images"] = X_test.shape[0]
    stats["image_size"] = X_train[0].shape
    return X_train, X_test, y_train, y_test, stats

def get_modified_model(model):
     #Transfer Learning - Tuning, weights will start from last check point
    # Extracting the input and second last layers of MobileNetv2
    base_input = model.layers[0].input
    base_ouput = model.layers[-2].output

    final_output = layers.Dense(128)(base_ouput) ## adding new layer after the output of  global pooling layer
    final_ouput = layers.Activation('relu')(final_output) ## activation function
    final_output = layers.Dense(64)(final_ouput)
    final_ouput = layers.Activation('relu')(final_output)
    final_output = layers.Dense(len(CONFIG.MOBILENET_CLASSES), activation='softmax')(final_ouput) ## my classes are seven
    model = keras.Model(inputs=base_input , outputs= final_output)

    # compile the modified model for training
    model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    return model


if __name__ == "__main__":

    print("loading dataset into memory...")
    # load dataset
    X_train, X_test, y_train, y_test, stats = load_dataset()
    # print the stats
    print("\n\n################ Dataset Summary ################")
    print(f"total training images loaded: {stats['training_images']}")
    print(f"total testing images loaded: {stats['testing_images']}")
    print(f"images size: {stats['image_size']}")
    print(f'\nCorrectly Masked Face: {stats["Correctly Masked Face"]} images')
    print(f'Incorrectly Masked Face: {stats["Incorrectly Masked Face"]} images')
    print(f'Unmasked Face: {stats["Unmasked Face"]} images')

    print("\n\nplotting dataset samples...")
    # plot samples
    systemRandom = random.SystemRandom()
    # initialize the plot
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(22, 14))
    for row in range(3):
        for col in range(5):
            # generate a random index
            index = systemRandom.randint(0, len(X_train))
            # select image based on the index and visualize
            axes[row][col].imshow(X_train[index], interpolation='nearest', aspect='auto')
    plt.show()

    print("\n\nbuilding wheel for MobileNetv2 model...")
    # initializing the MobileNet v2 model. If you are using it for the first time, the model weights will be downloaded
    # automatically.
    model = tf.keras.applications.MobileNetV2() ## pre-trained Model

    #Transfer Learning - Tuning, weights will start from last check point
    # Extracting the input and second last layers of MobileNetv2
    model = get_modified_model(model)
    model.summary()

    # set cutom callbacks
    # save the model on every epoch if accuracy improves
    cb_checkpt = ModelCheckpoint(CONFIG.MOBILENET_MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    # Reduce the learning rate for better training and fine tuning
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, mode='min')

    # Stop the training if accuracy is not improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)

    print("everythings set; starting training...")
    # start the training process
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=CONFIG.BATCH_SIZE,
        epochs=CONFIG.EPOCHS,
        shuffle=True,
        callbacks=[reduce_lr_loss, cb_checkpt, early_stopping],
        verbose=2
    )

    # save model after training
    model.save(CONFIG.MOBILENET_MODEL_PATH)
    print(f"training ended; trained model has been saved to {CONFIG.MOBILENET_MODEL_PATH}")

    print("evaluating model; plotting predictions...")
    # load the saved model
    model_loaded = keras.models.load_model(CONFIG.MOBILENET_MODEL_PATH)

    # plot samples
    systemRandom = random.SystemRandom()
    print("Total images in dataset: {}".format(len(X_test)))
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(24, 24))
    for row in range(4):
        for col in range(4):
            index = systemRandom.randint(0, len(X_test))
            img_batch = np.expand_dims(np.array(X_test[index]), axis=0)
            prediction = model_loaded.predict(img_batch)[0]
            prediction = CONFIG.MOBILENET_CLASSES[np.argmax(prediction)]
            axes[row][col].imshow(X_test[index], interpolation='nearest', aspect='auto')
            axes[row][col].title.set_text(prediction)
    plt.show()
    print("all done...")



    