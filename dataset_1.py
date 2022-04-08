import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence): 
  def read_from_file(directory, input_measure):
    x_return = []
    min_value = 50000.0
    max_value = -500000.0
    for filename in os.listdir(directory):
      x_array = []
      os.chdir(directory)
      label = filename.split('_')[-3]
      measure = filename.split('_')[-2]
      if input_measure == "All": 
        adding =  os.path.join(directory, filename)
        x_return.append(adding)
        with open(filename) as f:
          lines = f.readlines()
          for line in lines:
            line = line.replace("\n", '')
            if float(min_value)>float(line):
              min_value = line
            if float(max_value)<float(line):
              max_value = line
      elif input_measure == measure:
        adding =  os.path.join(directory, filename)
        x_return.append(adding)
        with open(filename) as f:
          lines = f.readlines()
          for line in lines:
            line = line.replace("\n", '')
            if float(min_value)>float(line):
              min_value = line        
            if float(max_value)<float(line):
              max_value = line
      os.chdir('../../')
    #x_return = np.array(x_return)
    #y_return = np.array(y_return)
    #exit()
    print (min_value)
    print (max_value)
    return x_return, min_value, max_value



  def normalize(input_array):
    min_value = np.min(input_array)
    max_value = np.max(input_array)
    normalized_array = []
    for x in input_array:
      normalized_value = ((x-min_value)/(max_value-min_value))
      normalized_array.append(normalized_value)
    return normalized_array

  def __init__(self, min_value, max_value, list_IDs, labels, batch_size, n_classes, n_channels, shuffle, ):
    self.n_channels = n_channels
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.min_value = min_value
    self.max_value = max_value
    self.on_epoch_end()    
    #print("inside init")
    #self.root = root
    #self.batch_size = batch_size
  
  def getAllFiles(self):
    pass


  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)

    return X, y


  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)


  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    #X = np.empty((self.batch_size, self.n_channels))
    X = []
    #y = np.empty((self.batch_size), dtype=int)
    Y = []
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      correct_ID = ID.replace("\\", "/")
      # Store class
      x_array = []
      with open(correct_ID) as f:
          lines = f.readlines()
          for line in lines:
            line = line.replace("\n", '')
            x_array.append(float(line))
          difference = 1000-len(x_array)
          for i in range(0,difference):
            x_array.append(0.0)


      min_value = float(self.min_value)
      max_value = float(self.max_value)
      normalized_array = []
      for x in x_array:
        normalized_value = ((x-min_value)/(max_value-min_value))*400
        normalized_array.append(normalized_value)
      # print (normalized_array)
    
      X.append(normalized_array)
      Y.append(self.labels[ID])

    #print ("shape of X: " , np.shape(X))
    #return X[:, :, np.newaxis], tf.keras.utils.to_categorical(Y, num_classes=self.n_classes)
    X = np.array(X)
    Y = np.array(Y)
    #print (X)
    # print (normalize(X))
    # min_value = np.min(X)
    # max_value = np.max(X)
    # print ("min: ", min_value)
    # print ("max: ", max_value)
    # print(min(Y))
    # print(max(Y))
    return X, Y

