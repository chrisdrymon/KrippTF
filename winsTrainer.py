import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
import pickle
from tensorflow.keras import layers

os.environ['CUDA_VISIBLE_DEVICES'] = ''

df = pd.read_csv('C:\\Users\\chris\\Google Drive\\Python\\KrippHistory.csv')
df = df.drop(columns='Bucket1')

# Create the unique card dictionary
i = 1
cards = []
preNump = []
preLabels = []
cardDic = {}
while i < 31:
    cardNum = 'Card' + str(i)
    cards = cards + df[cardNum].tolist()
    i += 1
print('Total Cards:', len(cards))
noDups = list(set(cards))
noDupsLen = len(noDups)
print('Unique Cards', noDupsLen)

j = 0
for card in noDups:
    cardDic[card] = j
    j += 1

pickleOut = open('C:\\Users\\chris\\Google Drive\\Python\\winsDeckDic.pkl', 'wb')
pickle.dump(cardDic, pickleOut, pickle.HIGHEST_PROTOCOL)
pickleOut.close()

# The other dictionaries
classDict = {'Druid': 0, 'Hunter': 1, 'Mage': 2, 'Paladin': 3, 'Priest': 4, 'Rogue': 5, 'Shaman': 6, 'Warlock': 7,
             'Warrior': 8}
deckType = {'Aggro-Control': 0, 'Attrition': 1, 'Classic Aggro': 2, 'Classic Control': 3, 'Mid-Range': 4, 'Tempo': 5}
expansion = {'GvG': 0, 'BRM': 1, 'WOG': 2, 'Kara': 3, 'MSG': 4, 'Ungoro': 5, 'KFT': 6, 'KnC': 7, 'Woods': 8,
             'Boomsday': 9, 'Rumble': 10, 'RoS': 11}
bucketDict = {'Zero': 0, 'Low': 1, 'High': 2}

# Create the tensors
for row in df.itertuples():

    # Classes
    classTens = [0] * 9
    classTens[classDict[row[4]]] = 1

    # Archetype
    archTens = [0] * 6
    archTens[deckType[row[5]]] = 1

    # Expansion
    expanTens = [0] * 12
    expanTens[expansion[row[6]]] = 1

    # Normalizing the deck scores before adding them to the tensor.
    deckScore = ((row[7] - 50) / 30)

    # Cards
    cardTens = [0] * noDupsLen
    k = 8
    while k < 38:
        if row[k] is not np.nan:
            cardTens[cardDic[row[k]]] = row[k+30]
        k += 1

    # Combine data into a single tensor.
    combinedTens = classTens + archTens + expanTens
    combinedTens.append(deckScore)
    combinedTens = combinedTens + cardTens

    # Add that tensor to the list of tensors.
    preNump.append(combinedTens)

    # Convert labels to hots and add to the list of labels.
    label = row[68]
    preLabels.append(label)

print('Input', len(combinedTens))
trainData = np.array(preNump)
trainLabels = np.array(preLabels)


# Custom Callback that Saves Model and Stop Training
class ModelSaver(tf.keras.callbacks.Callback):
    veryBest = 1000

    def on_train_begin(self, logs=None):
        self.tempBest = 1000

    def on_epoch_end(self, epoch, logs=None):
        # Saves the best model
        if logs['val_mean_squared_error'] < logs['mean_squared_error']:
            worst = logs['mean_squared_error']
        else:
            worst = logs['val_mean_squared_error']
        if worst < self.tempBest:
            self.tempBest = worst
            if self.tempBest < self.veryBest:
                self.veryBest = self.tempBest
                self.model.save('C:\\Users\\chris\\Google Drive\\Python\\winsModel.h5', overwrite=True)
                print('\n\nModel saved at epoch', epoch, 'with', self.veryBest, 'MSE.\n')
        if logs['val_mean_squared_error'] - self.tempBest > 1 and epoch > 30:
            self.model.stop_training = True
            print('\n\nTraining stopped to overfitting at epoch', epoch, '\n')

    def on_train_end(self, logs=None):
        print('\nBest Model saved at', self.veryBest, 'MSE.\n')


modelSaver = ModelSaver()

count = 0
columnNames = ['Batch Size', 'L1 Nodes', 'L1 Regularization', 'L1 Dropout', 'L2 Nodes', 'L2 Regularization',
               'L2 Dropout', 'Learning Rate', 'Accuracy']

while count < 100:
    bs1 = random.randint(1, 9)
    bs2 = random.randint(0, 2)
    batchSize = random.randint(1, 500)
    l1Nodes = random.randint(40, 500)
    l1Reg1 = random.randint(0, 9)
    l1Reg2 = random.randint(-5, -1)
    l1Reg = l1Reg1*10**l1Reg2
    l1Dropout = random.randint(0, 80)/100
    l2Nodes = random.randint(4, 500)
    l2Dropout = random.randint(0, 80)/100
    l2Reg1 = random.randint(0, 9)
    l2Reg2 = random.randint(-5, -1)
    l2Reg = l2Reg1*10**l2Reg2
    lr1 = random.randint(1, 9)
    lr2 = random.randint(-6, -1)
    lr = lr1*10**lr2

    tf.keras.backend.clear_session()
    graph = tf.Graph()
    with tf.Session(graph=graph):
        model = tf.keras.Sequential([layers.Dense(l1Nodes, kernel_regularizer=tf.keras.regularizers.l2(l1Reg),
                                                  activation='relu', input_shape=(len(combinedTens),)),
                                     layers.Dropout(l1Dropout),
                                     layers.Dense(l2Nodes, kernel_regularizer=tf.keras.regularizers.l2(l2Reg),
                                                  activation='relu'),
                                     layers.Dropout(l2Dropout),
                                     layers.Dense(1)])

        optimizer = tf.keras.optimizers.RMSprop(lr)

        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])

        callBacks = [modelSaver, tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20,
                                                                  restore_best_weights=False)]

        model.fit(x=trainData, y=trainLabels, batch_size=batchSize, epochs=1000, callbacks=callBacks,
                  validation_split=0.2, shuffle=True)

    count += 1
