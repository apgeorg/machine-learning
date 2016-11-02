from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras import backend as K

data_dir = './flower_photos/'
# num. of train samples
nb_train_samples = 3670
# num. of validation samples
nb_validation_samples = 1000
# dimensions of our images
img_width, img_height = 299, 299
# num. of epochs
nb_epoch = 10
# num. of output classes
nb_classes = 5
train_batch_size = 32
validation_batch_size = 32

# create the base pre-trained model
inceptionV3_model = InceptionV3(weights='imagenet', include_top=False)
print ("Layers of Inception V3: %i" % (len(inceptionV3_model.layers)))

# add a global spatial average pooling layer
x = inceptionV3_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
# and a logistic layer -- we have 5 classes
predictions = Dense(nb_classes, activation='softmax')(x)

# this is the model we will train
model = Model(input=inceptionV3_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in inceptionV3_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=train_batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=validation_batch_size,
        class_mode='categorical')

# train the model on the new data for a few epochs
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=2,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(inceptionV3_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

