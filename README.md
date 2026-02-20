##Face Mask Detection and Face Recognition

Face Mask Detection had become very important to many organizations due to recent bad situations from COVID19 pandamic. It's of great importance to wear masks in puplic to help prevent the spread of Corona Virus accross the globe. However, some people behave irresponsive in the matter and they do not care to follow the instruction reagarding COVID. Hence, it has become necessary to build a solution which will detect whether a person is wearing a mask or not. In addition, it's a surplus to detect incorrectly worn masks.
This work uses a deep-learning based solution for detecting faces wearing masks in public place to prevent the spread of Coronavirus is presented. This work does not only rely on the detection of mask but it also tried to identify people not wearing face masks using Face recognition techniques.
Face Recognition is an application of modern technologies that is capable of identifying or verifying human faces in an RGB image captured by a digital camera. Since the need of modern identity systems, face recognition is becoming more and more important and is being used by many organizations all over the world. 

General Objectives
1.	To implement a web app that provides facilities to detect a person if he’s wearing a face mask, he’s wearing is in a wrong way or he’s without a mask. 
2.	To recognize a person’s face who is not wearing a face mask.

Specific Objectives 
1.	To design and develop a face mask detection and person face recognition system with an RGB camera installed in specific location. 
2.	To design, implement and evaluate a code that will be able to extract frames from a camera and use Machine Learning Algorithms.
3.	To develop and implement a deep learning-based algorithm to follow the flow below:
a.	First detect faces in an image with an AP of at least 0.7.
b.	Then classify a face into masked, unmasked and wrongly masked face with an AP of at least 0.7
c.	Finally, recognize the face if it has been classified as unmasked with an AP of at least 0.7

Deep Learning Algorithms to be Used
There are many deep learning algorithms used for different purposes achieving human-level accuracies yet we need to choose wisely between them considering their pros and cons. Some of algorithms that are likely to be used are:
Face Detectors
1.	Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Network (MTCNN)
2.	RetinaFace: Single-stage Dense Face Localisation in the Wild

Classifiers:
1.	MobileNetV3 based classifier (images)
2.	MobileNetV2 based classifier (images)
3.	VGG16 (images)
4.	SVM (for face embeddings)

Face Embeddings Extractors:
1.	Google Facenet
2.	VGGFace

Updates:
Implementation Language: Python
Machine Learning frameworks used:
1.	Tensorflow 
2.	Keras
3.	Sklearn
Computer Vision Library used: OpenCV Python, PIL
Other Libraries Used: numpy, pandas, scipy, matplotlib

MobileNet Modification:
We are using MobileNetV2 for classification of masked face, unmasked face and incorrectly masked face classes. For training on our custom dataset, we are using pre-trained weights on ImageNet dataset which have 1000 classes. We are using transfer learning technique to reduce the training time and increase generalization (check transfer learning). For the modification step, we have extracted the features extraction part and MobileNetv2 head. The features extraction backbone will remain the same as it has been trained on ImageNet. Instead, we have modified the head by adding the following layers:
1.	Dense Layer with 128 filters and applied relu activation function
2.	Another Dense layer with 64 filters and applied relu activation function
3.	Another Dense layer with “num_classes” (3) filters and applied softmax activation function. The softmax is because we need the output probability for each class and then we will choose the one with highest probability.
The original backbone layer and the head were then recombined to make the final model ready for training. The model is then compiled with sparse_categorical_crossentropy loss, adam as optimizer and accuracy as metric to monitor.

We then trained the model for 30 epochs with custom callbacks applied which gave us an accuracy of above 95 percent on our test set. The train and test set has been split with ration of 80:20 percent.

To read more about MobileNetv2, visit the link below:
https://paperswithcode.com/method/mobilenetv2

Google FaceNet Algorithm:
Facenet is model released by Google for the extraction of facial features which returns a vector of 128 feature points for a single face. In our algorithm for face recognition, we are using Facenet model. The general steps involved in the process are below:
1.	First, we take a few images of a person e.g. 5 which is to be registered in the database. We extract facial features for each photo of the same person and we get 5 different vectors of 128 features for a single user. We do the same for every user which is get registered to generate a dataset of known users.
2.	We then store these feature vectors for every user in database along with the name of the user.
3.	In the third step, we train a classifier on these features which will be used to predict the person’s name. Every time, a user is registered, we first extract its facial features, then train the classifier and save the trained classifier for inference.
For the classification, we are using our own Keras classifier.
For the GUIs, we will be using Python frameworks like PyQt5 or Tkinter. We will decide which one fits our needs.
