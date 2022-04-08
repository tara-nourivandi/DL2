import os 
import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras import layers
from dataset_1 import DataGenerator


params = {'batch_size': 32,
          'n_classes': 10,
          'n_channels': 1,
          'shuffle': False}
numFeatures = 8
n_unique_words = 1000
timestep = 1000

def label_finder(labels, directory, input_measure):
	for filename in os.listdir(directory):
		os.chdir(directory)
		label = filename.split('_')[-3]
		measure = filename.split('_')[-2]
		if input_measure == "All":
			file_path = os.path.join(directory, filename)
			labels[file_path] =int(label)-1
		elif input_measure == measure:			
			file_path = os.path.join(directory, filename)
			labels[file_path] = int(label)-1
		os.chdir('../../')
	return labels
final_x = []

what_to_do = str(sys.argv[1])
input_directory = str(sys.argv[2])
model_name = str(sys.argv[3])
data_type = str(sys.argv[4])

# print ("w: ",what_to_do)
# print ("d: ",input_directory)
# print ("m: ",model_name)
# print ("t: ",data_type)
# exit()
if data_type == "EDA":
	input_measure = "EDA"
elif data_type == "mmhg":
	input_measure = "BP"
elif data_type == "mean":
	input_measure = "LA Mean BP"
elif data_type == "sys":
	input_measure = "LA Systolic BP"
elif data_type == "pulse":
	input_measure = "Pulse Rate"
elif data_type == "DIA":
	input_measure = "BP Dia"
elif data_type == "volt":
	input_measure = "Resp"
elif data_type == "resp":
	input_measure = "Respiration Rate"
elif data_type == "all":
	input_measure = "All"





if what_to_do == "train":
	#train_directory = os.path.join(input_directory, '/Training')
	train_directory = input_directory + '/Training'
	validation_directory = input_directory + '/Validation'
	# print (input_directory)
	# print (train_directory)
	# print (validation_directory)
	# exit()
	x_train, min_t, max_t = DataGenerator.read_from_file(train_directory, input_measure)
	# exit()
	x_validation, min_v, max_v = DataGenerator.read_from_file(validation_directory, input_measure)

	labels = {}
	labels = label_finder(labels, train_directory, input_measure)
	labels = label_finder(labels, validation_directory, input_measure)
	training_generator = DataGenerator(min_t, max_t, x_train, labels, **params)
	validation_generator = DataGenerator(min_v, max_v, x_validation, labels, **params)
	#exit()
	# norm_layer = layers.Normalization()
	# norm_layer.adapt(x_train)
	model = tf.keras.models.Sequential()
	#model.add(norm_layer)
	model.add(tf.keras.layers.Embedding(n_unique_words, numFeatures, input_length=timestep))
	model.add(tf.keras.layers.LSTM(32, return_sequences=True))
	model.add(tf.keras.layers.LSTM(64, return_sequences=True))
	model.add(tf.keras.layers.LSTM(128))
	model.add(tf.keras.layers.Dense(1024))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
	model.compile(loss='SparseCategoricalCrossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy']) 

	model.summary()


	model.fit_generator(generator=training_generator,
	                    validation_data=validation_generator, epochs = 10)