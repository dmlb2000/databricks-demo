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
file.extractall(join('/dbfs', 'mnt', 'databricks-dhdev-db-research'))

# COMMAND ----------


