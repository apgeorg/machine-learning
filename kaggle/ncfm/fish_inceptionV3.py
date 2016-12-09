"""
This example shows how to train "Inception V3" on "The Nature Conservancy Fisheries Monitoring" dataset.
The dataset is available on https://www.kaggle.com

Note that the following implementation does not provide a possible solution on 
"The Nature Conservancy Fisheries Monitoring" competition.  
"""
__author__ = "apgeorg"

# Fix random seed for reproducibility
import numpy as np
np.random.seed(911)

import os
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from collections import Counter
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model, load_model
from keras.layers import GlobalAveragePooling2D, AveragePooling2D 
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import __version__
from keras import backend as K
K.set_image_dim_ordering('th')

# Data directories
data_dir = "input/"
train_data_dir = data_dir + "train/train/"
test_data_dir = data_dir + "test_stg1/"

# Model directories
model_base = "netfiles/"
model_dir = "inceptionV3-kf15-run1/"
model_name = "fishy"

# Classes 
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
num_classes = len(classes)

# Dimensions of input images. The default image size for Inception V3 is 299x299.
img_width, img_height = 299, 299

def get_im_cv2(path):
    img = cv2.imread(path)
    resized_img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    #cv2.imwrite(path, resized_img)
    return resized_img

"""
Loads the train data. 
"""
def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    for folder in classes:
        idx = classes.index(folder)
        print("Load folder {} (Idx: {})".format(folder, idx))
        path = os.path.join(train_data_dir, folder)
        for fl in os.listdir(path):
            filename = os.path.basename(fl)
            img_path = os.path.join(path, filename)
            img = get_im_cv2(img_path)
            X_train.append(img)
            X_train_id.append(filename)
            y_train.append(idx)
    return X_train, y_train, X_train_id

"""
Load the test data.
"""
def load_test():
    path = test_data_dir
    data = []
    data_id = []
    for fl in sorted(os.listdir(path)):
        filename = os.path.basename(fl)
        img_path = os.path.join(path, filename)
        img = get_im_cv2(img_path)
        data.append(img)
        data_id.append(filename)
    return data, data_id

"""
Normalize input. 
"""
def normalize(x):
    # Convert to numpy
    x = np.array(x, dtype=np.uint8)
    # Reshape
    x = x.transpose((0, 3, 1, 2))
    # Convert to float.
    x = x.astype('float32')
    x = x / 255
    print("Shape:", x.shape)
    print(x.shape[0], " samples")
    return x


def get_train():
    # Load train
    X_train, y_train, X_train_id = load_train()
    # Normalize
    X_train = normalize(X_train)
    # Convert to numpy
    y_train = np.array(y_train, dtype=np.uint8)
    return X_train, y_train, X_train_id
    
def get_test():
    # Load test
    X_test, X_test_id = load_test()
    # Normalize
    X_test = normalize(X_test)
    return X_test, X_test_id

"""
Create submission file. 
""" 
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=classes)
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def print_distribution(y, tr, te): 
    print ("Distribution of Train")
    for key, val in Counter(y[tr]).items():
        print("{}: {}".format(classes[key], val))
    print ("\nDistribution of Test")
    for key, val in Counter(y[te]).items():
        print("{}: {}".format(classes[key], val))

def prepare_data_augmentation():
    datagen = ImageDataGenerator(
    rescale=1.2, 
    rotation_range=8., 
    shear_range=0.2, 
    zoom_range=0.3, 
    horizontal_flip=True, 
    vertical_flip=True)
    return datagen       
       
def load_inceptionV3():
    # Load pre-trained Model (Inception V3) with the pre-trained weights on Imagenet.
    # Don't include the 3 fully-connected layers on the top.     
    model = InceptionV3(weights='imagenet', include_top=False)  
    return model

def add_top_layers(base_model):
    # Adding some more layers on top which are randomly initialized. 
    x = GlobalAveragePooling2D()(base_model.output)
    #x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(base_model.output)
    #x = Dense(24, activation='relu')(x)
    y_pred = Dense(num_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=y_pred)
    return model

def freeze_all(model):
    # Freeze all pre-trained layers to train only the new top layers. 
    for layer in model.layers:
        layer.trainable = False

def create_model():
     # Load Inception V3
    inceptV3 = load_inceptionV3()
    
    # Freeze all layers
    freeze_all(inceptV3)
    
    # Add top layers
    model = add_top_layers(inceptV3)
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    """s
    # To get better results we can adapt some more layers in the Inception V3 model e.g. convolutional layers etc.
    # So we will freeze the bottom N layers and train the remaining top layers.
    N = 172 
    for layer in model.layers[:N]:
        layer.trainable = False
    for layer in model.layers[N:]:
        layer.trainable = True
    """
    return model

def run_cv_training(nfolds=10, bs=16, num_epochs=20):
    # Load train
    train_data, train_target, train_id = get_train()
    y_tmp = train_target # y_tmp contains not categorical class values
    train_target = np_utils.to_categorical(train_target, num_classes)
    #kfold = KFold(n_splits=nfolds, shuffle=True, random_state=51)
    #kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=51)
    kfold = ShuffleSplit(n_splits=nfolds, test_size=.2, random_state=51)
    # Data Augmetation 
    #train_datagen = prepare_data_augmentation()
    scores = []
    models = []
    #base_model = create_model()
    base_model = load_model(model_base + "fishy-fold-1-02-0.538.hdf5")
    # Start Kfold training 
    for k, (tr_idx, te_idx) in enumerate(kfold.split(train_data, y_tmp)):
        start_time = time.time()
        print("+++++++++++++++++++++++++++++++++++++++++++")
        print("Fold {} of {}:".format(k+1, nfolds)) 
        # Print the distribution
        #print_distribution(y_tmp, tr_idx, te_idx)
        # Creates the Model
        model = base_model
        # Get all train, test data & labels of the corresponding indexes 
        trX, teX = train_data[tr_idx], train_data[te_idx] 
        trY, teY = train_target[tr_idx], train_target[te_idx]
        print("Split in (train, test): ({}, {})".format(len(trX), len(teX)))
          
        # Callback
        earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
        modelcheckpoint = ModelCheckpoint(model_base + model_dir + model_name + "-fold-" + str(k+1) + "-{epoch:02d}-{val_loss:.3f}.hdf5", monitor='val_loss', save_best_only=True) 
        
        # Fits the model on batches with real-time data augmentation
        """
        model.fit_generator(train_datagen.flow(trX, trY, batch_size=bs), 
        samples_per_epoch=len(trX), 
        nb_epoch=num_epochs, 
        verbose=2, 
        validation_data=(teX, teY), 
        callbacks=[modelcheckpoint, earlystopping])
        """
        # Fits the model on batches
        model.fit(trX, trY, batch_size=bs, nb_epoch=num_epochs, shuffle=True, verbose=1, validation_data=(teX, teY), callbacks=[earlystopping, modelcheckpoint])

        # Evaluate model with validation data 
        pred = model.predict(teX, batch_size=bs, verbose=0)
        score = model.evaluate(teX, teY, verbose=0)
        # Add score to score list
        scores.append(score)
        # Add model to model list 
        models.append(model)
        print("Fold {} finished in {} sec".format(k+1, round(time.time()-start_time, 2)))
        print("+++++++++++++++++++++++++++++++++++++++++++")
    
    # Calculate mean and std of scores    
    print("\nCV log_loss: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
    info = 'loss_' + str(np.mean(scores)) + '_folds_' + str(nfolds) + '_ep_' + str(num_epochs)
    return info, models

""" Main """
if __name__ == "__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Competition: The Nature Conservancy Fisheries Monitoring")
    print("Author: ", __author__)
    print("Keras version: {}".format(__version__))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
    # Model parameters
    num_folds = 1
    batch_size = 32
    num_epochs = 2
    
    # Run CV 
    info, models = run_cv_training(num_folds, batch_size, num_epochs)
    # Run test process.
    test_res, test_id = run_testing(models, batch_size)
    # Create Submission.
    create_submission(test_res, test_id, info)
    

