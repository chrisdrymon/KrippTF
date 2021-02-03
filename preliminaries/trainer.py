from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from tensorflow.keras import layers, callbacks
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Preparing dictionaries to convert data into integers. Later they will be turned to one-hots.

classDict = {'Druid': 0, 'Hunter': 1, 'Mage': 2, 'Paladin': 3, 'Priest': 4, 'Rogue': 5, 'Shaman': 6,
             'Warlock': 7, 'Warrior': 8}
deckType = {'Aggro-Control': 0, 'Attrition': 1, 'Classic Aggro': 2, 'Classic Control': 3, 'Mid-Range': 4, 'Tempo': 5}
expansion = {'Vanilla': 0, 'BRM': 1, 'WOG': 2, 'Kara': 3, 'MSG': 4, 'Ungoro': 5, 'KFT': 6, 'KnC': 7, 'Woods': 8,
             'Boomsday': 9, 'Rumble': 10, 'RoS': 11}

df = pd.read_csv(os.path.join('data', 'KrippDateless.csv'), sep=',', header=None)
df = df.sample(frac=1)

preNump = []
preLabels = []
best = 0

for row in df.itertuples():

    # Turning classes to hots.
    hotNum = classDict[row[1]]
    classTens = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    classTens[hotNum] = 1

    # Normalizing the deck scores before adding them to the tensor.
    deckScore = ((row[2]-50)/30)

    # Turn deck type into a hot.
    deckHot = [0]*6
    deckHot[deckType[row[3]]] = 1

    # Turn expansion into a hot.
    expansHot = [0]*12
    expansHot[expansion[row[4]]] = 1

    # Combine data them into a single tensor.
    classTens.append(deckScore)
    combinedTens = classTens + deckHot + expansHot

    # Add that tensor to the list of tensors.
    preNump.append(combinedTens)

    # Convert that label to a hot and add to the list of labels.
    labelHot = [0, 0, 0]
    labelHot[row[5]] = 1
    preLabels.append(labelHot)


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    best = 0

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if logs['val_acc'] > logs['acc']:
            worst = logs['acc']
        else:
            worst = logs['val_acc']
        if worst > self.best:
            self.best = worst
            self.model.save(os.path.join('models', 'KrippModel.h5'), overwrite=True)
            print('Model saved at epoch', epoch, 'with', self.best, 'accuracy.')


def runnn(fcbk, prenump, prelabels):

    splitnum = int(df.shape[0] * .8)
    ftraindata = np.array(prenump[:splitnum])
    fevaldata = np.array(prenump[splitnum:])
    ftrainlabels = np.array(prelabels[:splitnum])
    fevallabels = np.array(prelabels[splitnum:])

    fdense1 = random.randint(4, 200)
    factivation1 = 'relu'
    fdropout1 = 0
    reg11 = random.randint(1, 9)
    reg12 = random.randint(-5, -3)
    freg1 = reg11*10**reg12
    fdense2 = random.randint(4, 200)
    factivation2 = 'relu'
    fdropout2 = random.randint(0, 80)/100
    reg21 = random.randint(1, 9)
    reg22 = random.randint(-5, -3)
    freg2 = reg21*10**reg22
    fbatchsize = random.randint(4, 300)
    lr1 = random.randint(1, 9)
    lr2 = random.randint(-4, -2)
    flearningrate = lr1*10**lr2

    fmodel = tf.keras.Sequential([layers.Dense(fdense1, kernel_regularizer=tf.keras.regularizers.l2(freg1),
                                               activation=factivation1, input_shape=(28,)),
                                 layers.Dropout(fdropout1),
                                 layers.Dense(fdense2, activation=factivation2),
                                 layers.Dropout(fdropout2),
                                 layers.Dense(3, activation='softmax')])

    optimizer = tf.keras.optimizers.Adam(lr=flearningrate)

    fmodel.compile(optimizer=optimizer,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    fcallbacks = [callbacks.EarlyStopping(monitor='val_acc', patience=50), fcbk]

    fthemodel = fmodel.fit(x=ftraindata, y=ftrainlabels, batch_size=fbatchsize, epochs=400, verbose=0,
                           callbacks=fcallbacks, validation_data=(fevaldata, fevallabels), shuffle=True,
                           initial_epoch=0)

    return fdense1, fdropout1, freg1, fdense2, fdropout2, freg2, fbatchsize, flearningrate, fthemodel


cbk = CustomModelCheckpoint()

theBest = 0
samples = []

i = 1
while i < 51:
    graph = tf.Graph()
    with tf.Session(graph=graph):
        dense1, dropout1, reg1, dense2, dropout2, reg2, batchSize, learningRate, theModel = runnn(cbk, preNump,
                                                                                                  preLabels)
        print(i, dense1, dropout1, reg1, dense2, dropout2, reg2, batchSize, learningRate)
        item = 0
        tempBest = 0
        while item < len(theModel.history['acc']):
            if theModel.history['acc'][item] < theModel.history['val_acc'][item]:
                theWorst = theModel.history['acc'][item]
            else:
                theWorst = theModel.history['val_acc'][item]
            if theWorst > theBest:
                theBest = theWorst
                bestModel = i
                bestDense1 = dense1
                bestDropout1 = dropout1
                bestReg1 = reg1
                bestDense2 = dense2
                bestDropout2 = dropout2
                bestReg2 = reg2
                bestBatchSize = batchSize
                bestLearningRate = learningRate
                bestEpoch = item + 1
            if theWorst > tempBest:
                tempBest = theWorst
            item += 1
    tempParameters = [dense1, dropout1, reg1, dense2, dropout2, reg2, batchSize, learningRate, tempBest]
    samples.append(tempParameters)
    i += 1
hyperParameters = np.asarray(samples)
np.savetxt(os.path.join('data', 'datelessparams.csv'), hyperParameters, delimiter=',')
print("Best Model:", bestModel)
print("Layer 1:", bestDense1, "nodes,", bestDropout1 * 100, "% dropout.")
print("Layer 2:", bestDense2, "nodes,", bestDropout2 * 100, "% dropout.")
print("Batch Size:", bestBatchSize)
print("Learning Rate:", bestLearningRate)
print('The best accuracy is', theBest * 100, '% at epoch', bestEpoch)
