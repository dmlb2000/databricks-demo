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


# COMMAND ----------


