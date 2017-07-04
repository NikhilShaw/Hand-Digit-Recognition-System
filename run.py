from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

########## VISUALIZING AND DATA PREPERATION ###########

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# plotting images for visualization
for x in range(0,4):
    plt.subplot(221+x)
    plt.imshow(x_train[0])
plt.show()
# reshape to be [samples][width][height]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# convert from int to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train/=255
x_test/=255
# define data preparation
train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, width_shift_range=0.2, height_shift_range=0.2 )
test_datagen = ImageDataGenerator()
# fit parameters from data
train_datagen.fit(x_train)
test_datagen.fit(x_test)
# configure batch size and retrieve one batch of images
for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=9):
# create a grid of 3x3 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
# show the plot
    plt.show()
    break

# one hot encoding the output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
no_of_class = y_test.shape[1]

############# MODEL ###########

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(no_of_class, activation='softmax'))
    # compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

new_model=create_model()
#checkpointing
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor= 'val_acc' , verbose=1, save_best_only=True,mode= 'max' )
callbacks_list = [checkpoint]
# fitting data
new_model.fit_generator(train_datagen.flow(x_train, y_train), samples_per_epoch= len(x_train), nb_epoch=10, callbacks= callbacks_list, validation_data= test_datagen.flow(x_test, y_test), validation_steps=len(x_test))   

















