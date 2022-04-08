import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
import numpy as np

class CustomClass(Sequence): 
  def read_from_file(self, directory, input_measure):
    x_return = []
    y_return = []

    for filename in os.listdir(directory):
      x_array = []
      os.chdir(directory)
      label = filename.split('_')[-3]
      measure = filename.split('_')[-2]
      if input_measure == "All":      
        with open(filename) as f:
          lines = f.readlines()
          for line in lines:
            line = line.replace("\n", '')
            x_array.append(float(line))
          y_return.append(int(label)-1)
          x_return.append(x_array)

      elif input_measure == measure:
        with open(filename) as f:
          lines = f.readlines()
          for line in lines:
            line = line.replace("\n", '')
            x_array.append(float(line))
          y_return.append(int(label)-1)
          x_return.append(x_array)
          difference = 1000-len(x_array)
          for i in range(0,difference):
            x_array.append(0.0)
      os.chdir('../../')
    x_return = np.array(x_return)
    y_return = np.array(y_return)
    return x_return, y_return
  def __init__(self):
    #print("inside init")
    #self.root = root
    #self.batch_size = batch_size
    pass
  def getAllFiles(self):
    pass


  def __getitem__(self, directory, validate_directory, input_measure):

    x_train = []
    y_train = []

    x_validate = []
    y_validate = []

    #print ("directory: " + directory)
    #print("before: " + os.getcwd())
    x_train, y_train = self.read_from_file(directory, input_measure)
    x_validate, y_validate = self.read_from_file(validate_directory, input_measure)

    #print (x_validate.shape)

    return x_train, y_train, x_validate, y_validate
  def __len__(self):
    pass

