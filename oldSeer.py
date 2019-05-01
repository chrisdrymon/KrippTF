from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pickle
from bs4 import BeautifulSoup
import requests
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load the model, card dictionary, and address
model = tf.keras.models.load_model('C:\\Users\\chris\\Google Drive\\Python\\OModel5316.h5')
pickleIn = open('C:\\Users\\chris\\Google Drive\\Python\\ODict5316.pkl', 'rb')
cardDict = pickle.load(pickleIn)
address = 'https://www.heartharena.com/arena-run/o6n42d'
lettuce = 1163

# Preparing dictionaries to convert data into integers. Later they will be turned to one-hots.
classDict = {'Druid': 0, 'Hunter': 1, 'Mage': 2, 'Paladin': 3, 'Priest': 4, 'Rogue': 5, 'Shaman': 6,
             'Warlock': 7, 'Warrior': 8}
deckType = {'Aggro-Control': 0, 'Attrition': 1, 'Classic Aggro': 2, 'Classic Control': 3, 'Mid-Range': 4, 'Tempo': 5}
expansion = {'Vanilla': 0, 'BRM': 1, 'WOG': 2, 'Kara': 3, 'MSG': 4, 'Ungoro': 5, 'KFT': 6, 'KnC': 7, 'Woods': 8,
             'Boomsday': 9, 'Rumble': 10, 'RoS': 11}

# Load deck info from address
source = requests.get(address)
newSoup = BeautifulSoup(source.text, 'lxml')
cardList = newSoup.find('ul', class_='deckList')
gameList = newSoup.find('ul', class_='matches-list')

# Class
hsClass = newSoup.title.text.split()[0]
classTens = [0] * 9
classTens[classDict[hsClass]] = 1

# Archetype
archetype = newSoup.find('header', class_='deck-archetype-name').find('span')
archetype = str(archetype)
archetype = archetype[6:]
archetype = archetype.split('<br/>')[0]
archTens = [0] * 6
archTens[deckType[archetype]] = 1

# Expansion
expanTens = [0] * 12
expanTens[expansion['RoS']] = 1

# Deck Score
score = newSoup.find('section', class_='arenaDeckTierscore').find('span').text
deckScore = (float(score)-50)/30

# Cards in Deck
deckList = []
for card in cardList.find_all('span', class_='name'):
    deckList.append(card.text)

# Quantity of Each Card in Deck
quantList = []
for card in cardList.find_all('span', class_='quantity'):
    quantList.append(card.text)

# Form the card tensor
cardTens = [0]*len(cardDict)
i = 0
while i < len(deckList):
    try:
        cardTens[cardDict[deckList[i]]] = quantList[i]
    except KeyError:
        print('One card missing.')
        pass
    i += 1

# Combine data them into a single tensor.
combinedTens = classTens + archTens + expanTens
combinedTens.append(deckScore)
combinedTens = combinedTens + cardTens

predicTens = [[combinedTens]]
prediction = model.predict(predicTens)
bet = (max(prediction[0])-.5)*2*lettuce
print(archetype, hsClass, score)
print(prediction[0])
print("Bet", int(bet), "lettuce.")
