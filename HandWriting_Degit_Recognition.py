import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
import random

np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

sns.set(style='white', context='notebook', palette='deep')

train=pd.read_csv("train.csv")

test=pd.read_csv("test.csv")

Y_train=train["label"]
X_train=train.drop(labels=["label"],axis=1)

sns.countplot(Y_train)

X_train.isnull().any().describe()

test.isnull().any().describe()

X_train = X_train/255.
test = test/255.

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train, num_classes = 10)

X_train,X_val,Y_train,Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)

print(f"Training shape {X_train.shape}\nValidation shape {X_val.shape}")

g = plt.imshow(X_train[1][:,:,0])

training = np.array(train, dtype = 'float32')
testing = np.array(test, dtype='float32')

# Let's view some images!
i = random.randint(1,37800) # select any random index from 1 to 60,000
plt.imshow(training[i,1:].reshape((28,28)) ) # reshape and plot the image

plt.imshow(training[i,1:].reshape((28,28)) , cmap = 'gray') # reshape and plot the image

# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 10
L_grid = 10

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(W_grid,L_grid, figsize = (17,17))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_training = len(training) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index    
    axes[i].imshow( training[index,1:].reshape((28,28)) )
    axes[i].set_title(training[index,0], fontsize = 8)# 0 is for label column
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


#import tensorflow as tf
"""from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense"""

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu',))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))          
model.add(Dense(10, activation='softmax'))


from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
from IPython.display import Image
Image('model.png')

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                           patience=3,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=0.00001)




datagen = ImageDataGenerator(
featurewise_center=False,
samplewise_center=False,
featurewise_std_normalization=False,
samplewise_std_normalization=False,
zca_whitening=False,
rotation_range=10,
zoom_range=0.1,
width_shift_range=0.1,
horizontal_flip=False,
vertical_flip=False)

datagen.fit(X_train)

epochs = 30
batch_size = 128

epochs = 30
batch_size = 128


# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


Results = model.predict(test)

Results = np.argmax(Results,axis = 1)

Results = pd.Series(Results,name="Label")

print(Results)

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),Results],axis = 1)

print(submission)

