"""
This example shows how to train Inception V3 on a flower dataset.
The dataset should be manually seperated into a train and validation set. 
You can download it on your own from https://www.tensorflow.org/versions/r0.9/how_tos/image_retraining/index.html
by following the instructions or you can run the "install.sh" script. 
"""

# Fix random seed for reproducibility
import numpy as np
np.random.seed(7)

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D  
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import sys
sys.path.append("/home/apgeorg/projects/github/public/keras-lib/")
from utils.models import print_layers

# Data directories
data_dir = "flower_photos/"
train_data_dir = data_dir + "train/"
validation_data_dir = data_dir + "validation/"

# Model parameters
num_epoch = 2
num_class = 5
train_batch_size = 32
validation_batch_size = 32

# Dimensions of input images. The default image size for Inception V3 is 299x299.
img_width, img_height = 299, 299

# Load pre-trained Model (Inception V3) with the pre-trained weights on Imagenet.
# Don't include the 3 fully-connected layers on the top.     
inceptV3_model = InceptionV3(weights='imagenet', include_top=False)

# Adding some more layers on top which are randomly initialized. 
x = GlobalAveragePooling2D()(inceptV3_model.output)
x = Dense(2048, activation='relu')(x)
y_pred = Dense(num_class, activation='softmax')(x)
model = Model(input=inceptV3_model.input, output=y_pred)

# Freeze all pre-trained layers to train only the new top layers.  
for layer in inceptV3_model.layers:
    layer.trainable = False
   
print_layers(model)

# RMSprop used as optimizer
rmsprop = RMSprop(lr=0.001)
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

# Configure data augmentation for training.  
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False)

# Configure data augmentation for testing.
test_datagen = ImageDataGenerator(rescale=1./255)

# Generates batches of augmented data from the directory. Data are shuffled with random seed.  
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=train_batch_size,
        seed=1,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=validation_batch_size,
        seed=2,
        class_mode='categorical')
      
# Callback for saving weights after the end of an epoch.
filename="flowers-inceptionV3-top-{epoch:02d}-{val_acc:.2f}.train.hdf5"
cb_save_best_after_epoch = ModelCheckpoint(filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Train the model for a few epochs.
# Fits the model on data-generated batch-by-batch.
model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_generator.filenames),
        nb_epoch=num_epoch,
        validation_data=validation_generator,
        nb_val_samples=1024,
        callbacks=[cb_save_best_after_epoch])       

# To get better results we can adapt some more layers in the Inception V3 model e.g. convolutional layers etc.
# So we will freeze the bottom N layers and train the remaining top layers.
N = 172
for layer in model.layers[:N]:
   layer.trainable = False
for layer in model.layers[N:]:
   layer.trainable = True
   
# Recompile the model for these modifications to take effect.
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Set a different filename
filename="flowers-inceptionV3-172-{epoch:02d}-{val_acc:.2f}.train.hdf5"   

# Retrain the model 
model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_generator.filenames),
        nb_epoch=num_epoch,
        validation_data=validation_generator,
        nb_val_samples=1024,
        callbacks=[cb_save_best_after_epoch])
        
        




