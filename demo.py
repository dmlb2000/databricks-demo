# Databricks notebook source
import requests
import tarfile
import os
from os.path import join

local_filename = "/FileStore/cifar-10-python.tar.gz"
with requests.get("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", stream=True, proxies=proxies) as r:
    r.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): 
            f.write(chunk)

file = tarfile.open(local_filename)
file.extractall(join('/FileStore', 'data'))

for root, dirs, files in os.walk(join('/FileStore', 'data', 'cifar-10-batches-py')):
    for filename in files:
        os.symlink(join('/FileStore', 'data', 'cifar-10-batches-py', filename), join('data', filename))
print("Done!")

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import numpy as np
# MAGIC import matplotlib
# MAGIC from matplotlib import pyplot as plt
# MAGIC from keras.models import Sequential
# MAGIC from keras.optimizer_v2.adam import Adam
# MAGIC from keras.callbacks import ModelCheckpoint
# MAGIC from keras.models import load_model
# MAGIC from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

# COMMAND ----------

from helper import get_class_names, get_train_data, get_test_data, plot_images, plot_model

# COMMAND ----------

matplotlib.style.use('ggplot')

# COMMAND ----------

class_names = get_class_names()
print(class_names)

# COMMAND ----------

num_classes = len(class_names)
print(num_classes)
