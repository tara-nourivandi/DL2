import numpy as np
import os
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import sys
import tensorflow.keras.backend as K
from keras import metrics 
import keras_metrics
from keras.datasets import imdb
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from keras import backend as kerasBackend
from sklearn.metrics import f1_score
from keras.utils.vis_utils import plot_model 
#from keras_visualizer import visualizer
#import visualkeras 

n_unique_words = 1000
timestep = 1000
numFeatures = 32
batchSize = 32

dataTypeNames = ["BP Dia_mmHg", "BP_mmHg", "EDA_microsiemens", "LA Mean BP_mmHg", 
              "LA Systolic BP_mmHg", "Pulse Rate_BPM", "Resp_Volts", "Respiration Rate_BPM"]
classesNames = ["1-Happy", "2-Sad", "3-Surprise", "4-Pain", "5-Disgust", "6-Afraid", "7-Startled", "8-Skeptical",
               "9-Embarrassment", "10-Fear"]
phases = ["Training", "Validation", "Testing"]
genderType = ["F", "M"]
filesFormat = ".txt"

phaseIndex = 0
directory = "./Physiological"
modelName = ""
dataTypeIndex = 0

x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [] ,[]

def getInputs():
    global phaseIndex, directory, modelName, dataTypeIndex

    if (sys.argv[1] == "train"):
        phaseIndex = 0
    elif (sys.argv[1] == "test"):
        phaseIndex = 2
    # elif (sys.argv[1] == "-trainBest"):
    #     phaseIndex = 0
    else:
        print ("Wrong phase input !")
        exit()

    directory = sys.argv[2]
    modelName = sys.argv[3]

    if (sys.argv[4] == "DIA"):
        dataTypeIndex = 0
    elif (sys.argv[4] == "mmhg"):
        dataTypeIndex = 1
    elif (sys.argv[4] == "EDA"):
        dataTypeIndex = 2
    elif (sys.argv[4] == "mean"):
        dataTypeIndex = 3
    elif (sys.argv[4] == "sys"):
        dataTypeIndex = 4
    elif (sys.argv[4] == "pulse"):
        dataTypeIndex = 5
    elif (sys.argv[4] == "volt"):
        dataTypeIndex = 6
    elif (sys.argv[4] == "resp"):
        dataTypeIndex = 7
    elif(sys.argv[4] == "all"):
        dataTypeIndex = 8       
    else:
        print ("Wrong data type input !")
        exit ()

    print ("Phase = ", phases[phaseIndex])
    print ("Directory = ", directory)
    print ("Model name = ", modelName)
    if (dataTypeIndex != 8):
        print ("Data type = ", dataTypeNames[dataTypeIndex])

    for gender in range(2):
        for i in range(100):
            for classIndex in range(10):
                if (dataTypeIndex != 8):
                    fileNameOrg = directory + "/" + phases[phaseIndex] + "/" + genderType[gender] + "{0:03}".format(i+1) + "_" + str(classIndex+1) + "_" + dataTypeNames[dataTypeIndex] + filesFormat
                    readFile(fileNameOrg, phases[phaseIndex], classIndex) 
                    if (phaseIndex == 0):
                        validationFileName = directory + "/" + phases[1] + "/" + genderType[gender] + "{0:03}".format(i+1) + "_" + str(classIndex+1) + "_" + dataTypeNames[dataTypeIndex] + filesFormat
                        readFile(validationFileName, phases[1], classIndex)
                else:
                    for dataTypeIter in range(8):
                        fileNameOrg = directory + "/" + phases[phaseIndex] + "/" + genderType[gender] + "{0:03}".format(i+1) + "_" + str(classIndex+1) + "_" + dataTypeNames[dataTypeIter] + filesFormat
                        readFile(fileNameOrg, phases[phaseIndex], classIndex)
                        if (phaseIndex == 0):
                            validationFileName = directory + "/" + phases[1] + "/" + genderType[gender] + "{0:03}".format(i+1) + "_" + str(classIndex+1) + "_" + dataTypeNames[dataTypeIter] + filesFormat
                            readFile(validationFileName, phases[1], classIndex)

def readFile(path, phase, label_index):
    global x_train, y_train, x_val, y_val, x_test, y_test
    try:
        with open(path, 'r') as File:
            infoFile = File.readlines() #Reading all the lines from File
            eachFile = []
            counter = 0
            for line in infoFile: #Reading line-by-line
                words = line.split()
                eachFile.append(float(words[0]))
                counter = counter + 1
                if (counter == timestep):
                    break

            while (counter < 1000):
                counter = counter + 1
                eachFile.append(float(0))
        if (phase == phases[0]): # Training
            x_train.append(eachFile)
            y_train.append(label_index)
        elif (phase == phases[1]): # Validation
            x_val.append(eachFile)
            y_val.append(label_index)
        elif (phase == phases[2]): # Testing
            x_test.append(eachFile)
            y_test.append(label_index)
    except FileNotFoundError as e:
        er = 1

def NormalizeData(data):
    maxVal = 0
    minVal = 10000
    for i in range(len(data)):
        if(np.max(data[i]) > maxVal):
            maxVal = np.max(data[i])
        if (np.min(data[i]) < minVal):
            minVal = np.min(data[i])
    for j in range(len(data)):
        data[j] = ((data[j] - minVal) / (maxVal - minVal)) * 100
    print (minVal)
    print (maxVal)
    exit()
    return data

    
def MakeModel():

    model = tf.keras.models.Sequential()
    #create Embedding layer - takes input and makes it easy to use with LSTM
    #can "manually" do this yourself if you don't want to use embedding
    #model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation = 'relu', padding='same', input_shape =(1000, 1000, 1)))
    model.add(tf.keras.layers.Embedding(n_unique_words, numFeatures, input_length=timestep))
    model.add(tf.keras.layers.Normalization())
    #uncomment link below for vanilla RNN
    #model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True))
    #model.add(tf.keras.layers.SimpleRNN(64))
    #uncomment line below for bidirectional LSTM
    #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)))
    #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    #uncomment line below to add another LSTM layer - return_sequences=True passes all hidden states to the next layer which is needed. Setting this to true is needed for all 
    #model.add(tf.keras.layers.LSTM(16, return_sequences=True))
    model.add(tf.keras.layers.LSTM(32, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(1024))
    #model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

    #configure model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001), #set optimizer to Adam; for now know that optimizers help minimize loss (how to change weights)
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), #sparce categorical cross entropy (measure predicted dist vs. actual)
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], #how often do predictions match labels
    )

    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model


def Train(name, epochsNum):
    model = MakeModel()
    model.summary()
    model.fit(x_train, y_train,
           batch_size=batchSize,
           epochs=epochsNum)
    model.save(name+".h5")
    print("Model saved.")

def TrainBest(name):
    checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath=name+".h5", 
                             monitor='loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min'),
                tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.01, patience=5)]
                
    model = MakeModel()
    model.summary()
    model.fit(x_train, y_train,
           batch_size=batchSize,
           epochs=25,
           validation_data=[x_val, y_val], 
           callbacks=checkpoint)

def Predict(model, testingData, testingLabels):
    #predict and format output to use with sklearn
    predict = model.predict(testingData)
    predict = np.argmax(predict, axis=1)
    #macro precision and recall
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(testingLabels, predict)
    precisionMacro = precision_score(testingLabels, predict, average='macro')
    recallMacro = recall_score(testingLabels, predict, average='macro')
    #micro precision and recall
    precisionMicro = precision_score(testingLabels, predict, average='micro')
    recallMicro = recall_score(testingLabels, predict, average='micro')
    confMat = confusion_matrix(testingLabels, predict)

    f1ScoreMacro = 2 * ((precisionMacro*recallMacro)/(precisionMacro+recallMacro))
    f1ScoreMicro = 2 * ((precisionMicro*recallMicro)/(precisionMicro+recallMicro))

    print("Accuracy: ", accuracy.result().numpy())
    print("Macro precision: ", precisionMacro)
    print("Micro precision: ", precisionMicro)
    print("Macro recall: ", recallMacro)
    print("Micro recall: ", recallMicro)
    print("Macro F1 Score: ", f1ScoreMacro)
    print("Micro F1 Score: ", f1ScoreMicro)
    print(confMat)

def Test(name):
    print("Loading Test Data")
    print("Loading model")
    model = tf.keras.models.load_model(name+".h5")
    model.summary()
    print("Making predictions on test data")
    Predict(model, x_test, y_test)

def main():
    if sys.argv[1] == "train":
    #     Train(modelName, 5)
    # elif sys.argv[1] == "trainBest":
        TrainBest(modelName)
    elif sys.argv[1] == "test":
        Test(modelName)

########################################################## Main Section ##########################################################
getInputs()
          
print ("Size x_train = ",  np.size(x_train))
print ("Size x_val = ",    np.size(x_val))
print ("Size x_test = ",   np.size(x_test))

x_train = NormalizeData(x_train)
x_val   = NormalizeData(x_val)
x_test  = NormalizeData(x_test)

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_val   = tf.convert_to_tensor(x_val,   dtype=tf.float32)
x_test  = tf.convert_to_tensor(x_test,  dtype=tf.float32)

y_train = np.array(y_train)
y_val   = np.array(y_val) 
y_test  = np.array(y_test)

# print ("x_train size = ", np.size(x_train))
# print ("y_train size = ", np.size(y_train))
# print ("Train shape= ", x_train.shape)
# print ("Label shape= ", y_train.shape)

main()

exit ()
