# Databricks notebook source
import requests
import tarfile
import os
from os.path import join

local_filename = "/tmp/cifar-10-python.tar.gz"
with requests.get("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", stream=True) as r:
    r.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): 
            f.write(chunk)

file = tarfile.open(local_filename)
file.extractall(join('/tmp', 'data'))

os.makedirs('data', exist_ok=True)
for root, dirs, files in os.walk(join('/tmp', 'data', 'cifar-10-batches-py')):
    for filename in files:
        linkpath = join('data', filename)
        if os.access(linkpath, os.R_OK):
            os.remove(linkpath)
        os.symlink(join('/tmp', 'data', 'cifar-10-batches-py', filename), linkpath)
print("Done!")

# COMMAND ----------

# MAGIC %pip install tensorflow

# COMMAND ----------

# MAGIC %fs ls file:/tmp/data/cifar-10-batches-py

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import numpy as np
# MAGIC import matplotlib
# MAGIC from matplotlib import pyplot as plt
# MAGIC from tensorflow.keras.models import Sequential
# MAGIC from tensorflow.keras.optimizers import Adam
# MAGIC from tensorflow.keras.callbacks import ModelCheckpoint
# MAGIC from tensorflow.keras.models import load_model
# MAGIC from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

# COMMAND ----------

# CIFAR - 10

# To decode the files
import pickle
# For array manipulations
import numpy as np
# To make one-hot vectors
from tensorflow.keras.utils import to_categorical
# To plot graphs and display images
from matplotlib import pyplot as plt


#constants

path = "data/"  # Path to data 

# Height or width of the images (32 x 32)
size = 32 

# 3 channels: Red, Green, Blue (RGB)
channels = 3  

# Number of classes
num_classes = 10 

# Each file contains 10000 images
image_batch = 10000 

# 5 training files
num_files_train = 5  

# Total number of training images
images_train = image_batch * num_files_train

# https://www.cs.toronto.edu/~kriz/cifar.html


def unpickle(file):  
    
    # Convert byte stream to object
    with open(path + file,'rb') as fo:
        print("Decoding file: %s" % (path+file))
        dict = pickle.load(fo, encoding='bytes')
       
    # Dictionary with images and labels
    return dict




def convert_images(raw_images):
    
    # Convert images to numpy arrays
    
    # Convert raw images to numpy array and normalize it
    raw = np.array(raw_images, dtype = float) / 255.0
    
    # Reshape to 4-dimensions - [image_number, channel, height, width]
    images = raw.reshape([-1, channels, size, size])

    images = images.transpose([0, 2, 3, 1])

    # 4D array - [image_number, height, width, channel]
    return images




def load_data(file):
    # Load file, unpickle it and return images with their labels
    
    data = unpickle(file)
    
    # Get raw images
    images_array = data[b'data']
    
    # Convert image
    images = convert_images(images_array)
    # Convert class number to numpy array
    labels = np.array(data[b'labels'])
        
    # Images and labels in np array form
    return images, labels




def get_test_data():
    # Load all test data
    
    images, labels = load_data(file = "test_batch")
    
    # Images, their labels and 
    # corresponding one-hot vectors in form of np arrays
    return images, labels, to_categorical(labels,num_classes)




def get_train_data():
    # Load all training data in 5 files
    
    # Pre-allocate arrays
    images = np.zeros(shape = [images_train, size, size, channels], dtype = float)
    labels = np.zeros(shape=[images_train],dtype = int)
    
    # Starting index of training dataset
    start = 0
    
    # For all 5 files
    for i in range(num_files_train):
        
        # Load images and labels
        images_batch, labels_batch = load_data(file = "data_batch_" + str(i+1))
        
        # Calculate end index for current batch
        end = start + image_batch
        
        # Store data to corresponding arrays
        images[start:end,:] = images_batch        
        labels[start:end] = labels_batch
        
        # Update starting index of next batch
        start = end
    
    # Images, their labels and 
    # corresponding one-hot vectors in form of np arrays
    return images, labels, to_categorical(labels,num_classes)
        


def get_class_names():

    # Load class names
    raw = unpickle("batches.meta")[b'label_names']

    # Convert from binary strings
    names = [x.decode('utf-8') for x in raw]

    # Class names
    return names



def plot_images(images, labels_true, class_names, labels_pred=None):

    assert len(images) == len(labels_true)

    # Create a figure with sub-plots
    fig, axes = plt.subplots(3, 3, figsize = (8,8))

    # Adjust the vertical spacing
    if labels_pred is None:
        hspace = 0.2
    else:
        hspace = 0.5
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Fix crash when less than 9 images
        if i < len(images):
            # Plot the image
            ax.imshow(images[i], interpolation='spline16')
            
            # Name of the true class
            labels_true_name = class_names[labels_true[i]]

            # Show true and predicted classes
            if labels_pred is None:
                xlabel = "True: "+labels_true_name
            else:
                # Name of the predicted class
                labels_pred_name = class_names[labels_pred[i]]

                xlabel = "True: "+labels_true_name+"\nPredicted: "+ labels_pred_name

            # Show the class on the x-axis
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Show the plot
    plt.show()
    

def plot_model(model_details):

    # Create sub-plots
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # Summarize history for accuracy
    #print(list(model_details.history.keys()))
    axs[0].plot(range(1,len(model_details.history['accuracy'])+1),model_details.history['accuracy'])
    axs[0].plot(range(1,len(model_details.history['val_accuracy'])+1),model_details.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['accuracy'])+1),len(model_details.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # Summarize history for loss
    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    
    # Show the plot
    plt.show()



def visualize_errors(images_test, labels_test, class_names, labels_pred, correct):
    
    incorrect = (correct == False)
    
    # Images of the test-set that have been incorrectly classified.
    images_error = images_test[incorrect]
    
    # Get predicted classes for those images
    labels_error = labels_pred[incorrect]

    # Get true classes for those images
    labels_true = labels_test[incorrect]
    
    
    # Plot the first 9 images.
    plot_images(images=images_error[0:9],
                labels_true=labels_true[0:9],
                class_names=class_names,
                labels_pred=labels_error[0:9])
    
    
def predict_classes(model, images_test, labels_test):
    
    # Predict class of image using model
    class_pred = model.predict(images_test, batch_size=32)

    # Convert vector to a label
    labels_pred = np.argmax(class_pred,axis=1)

    # Boolean array that tell if predicted label is the true label
    correct = (labels_pred == labels_test)

    # Array which tells if the prediction is correct or not
    # And predicted labels
    return correct, labels_pred


# COMMAND ----------

matplotlib.style.use('ggplot')

# COMMAND ----------

class_names = get_class_names()
print(class_names)

# COMMAND ----------

num_classes = len(class_names)
print(num_classes)

# COMMAND ----------

# Hight and width of the images
IMAGE_SIZE = 32
# 3 channels, Red, Green and Blue
CHANNELS = 3

# COMMAND ----------

images_train, labels_train, class_train = get_train_data()

# COMMAND ----------

print(labels_train)

# COMMAND ----------

print(class_train)

# COMMAND ----------

images_test, labels_test, class_test = get_test_data()

# COMMAND ----------

print("Training set size:\t",len(images_train))
print("Testing set size:\t",len(images_test))

# COMMAND ----------

def cnn_model():
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))    
    model.add(Conv2D(32, (3, 3), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    
    return model

# COMMAND ----------

model = cnn_model()

# COMMAND ----------

checkpoint = ModelCheckpoint('best_model_simple.h5',  # model filename
                             monitor='val_loss', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto') # The decision to overwrite model is made 
                                          # automatically depending on the quantity to monitor 

# COMMAND ----------

model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=1.0e-4), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) # Metrics to be evaluated by the model

# COMMAND ----------

model_details = model.fit(images_train, class_train,
                    batch_size = 128, # number of samples per gradient update
                    epochs = 100, # number of iterations
                    validation_data= (images_test, class_test),
                    callbacks=[checkpoint],
                    verbose=1)

# COMMAND ----------

scores = model.evaluate(images_test, class_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# COMMAND ----------

plot_model(model_details)

# COMMAND ----------

class_pred = model.predict(images_test, batch_size=32)
print(class_pred[0])

# COMMAND ----------

labels_pred = np.argmax(class_pred,axis=1)
print(labels_pred)

# COMMAND ----------

correct = (labels_pred == labels_test)
print(correct)
print("Number of correct predictions: %d" % sum(correct))

# COMMAND ----------

num_images = len(correct)
print("Accuracy: %.2f%%" % ((sum(correct)*100)/num_images))

# COMMAND ----------

incorrect = (correct == False)

# Images of the test-set that have been incorrectly classified.
images_error = images_test[incorrect]

# Get predicted classes for those images
labels_error = labels_pred[incorrect]

# Get true classes for those images
labels_true = labels_test[incorrect]

# COMMAND ----------

plot_images(images=images_error[0:9],
            labels_true=labels_true[0:9],
            class_names=class_names,
            labels_pred=labels_error[0:9])
