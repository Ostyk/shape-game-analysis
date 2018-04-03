from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import CSVLogger
import os
import matplotlib.pyplot as plt
from numpy import savetxt


# CNN MODEL
def Unit9(img_width, img_height): #n--numebr of classes
	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_width, img_height)
	else:
		input_shape = (img_width, img_height, 3)
	print(input_shape)
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3)) #3 classes
	model.add(Activation('sigmoid'))
	return model


#size of our generated images
img_width, img_height = 300, 300

model = Unit9(img_width, img_height)
#tracking our model
csv_logger = CSVLogger('training.log')

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])


#Get testing and training directories and number of files in all subfolders (diffrent class ex: squares etc.)
train_data_dir = 'data/train'
validation_data_dir = 'data/test'
#number of training images
nb_train_samples = 0
for root, dirs, files in os.walk(train_data_dir):
    nb_train_samples += len(files)
#number of images used for testing (validation)
nb_validation_samples = 0
for root, dirs, files in os.walk(validation_data_dir):
    nb_validation_samples += len(files)

epochs = 20
batch_size = 20

##Data augmentation##
#for training
train_datagen = ImageDataGenerator(
    rotation_range = 30,
    rescale=1. / 255,
    zoom_range=0.2,
    horizontal_flip=True)

#for testing
test_datagen = ImageDataGenerator(rescale=1. / 255)

#further augmentation of our data
#training
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode = 'categorical')
#testing
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode = 'categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
	callbacks=[csv_logger], #logging our progress
	verbose = 1)

try:
	scores = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size, pickle_safe = False)
	predict = model.predict_generator(validation_generator, nb_validation_samples // batch_size, verbose=1)
	savetxt('scores.txt', scores)
	savetxt('predictions.txt', predict)
except BaseException as error:
    print('An exception occurred: {}'.format(error))

model.save_weights('my_model_weights_2.h5') #saving weights for further analysis
model.save('my_model_2.h5')
