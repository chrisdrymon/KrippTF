"""
This module's purpose: The Hearthstone player Kripparrian in his Twitch channel would allow the betting of an
in-channel currency dubbed "lettuce." Upon him completing the drafting of a deck while playing arena mode, he gave his
tens of thousands of viewers a minute or two to wager their lettuce that he would get either 0-4 wins, 5-8 wins, or
9-12 wins with that deck in that arena run. Winners would have the amount of lettuce he bet doubled. Losers lost their
wager. This was a fun project for me to learn the Keras API while maximizing lettuce.

What this module does is take:

1) the amount of currency defined by the "lettuce" variable and
2) a drafted Hearthstone arena deck which is recorded at the URL assigned to the "address" variable.

It downloads that deck's cards and other information provided at the URL, processes it through a deep neural network,
which has been previously trained on the prior gameplay of Hearthstone player Kripparrian (about 1600 games), applies
the Kelly criterion according to the DNN's confidence level, and finally...

It prints win probabilities and optimal wagers.

* If I were to develop this further, I would go to the trouble of temperature scaling for more accurate probabilities.
Kripp, however, doesn't play arena anymore. And as is, it worked well enough to quickly catapult me to the top 10% of
lettuce holders before he stopped playing."""

import tensorflow as tf
import pickle
from bs4 import BeautifulSoup
import requests
import os
import numpy as np

# Enter amount of lettuce currently held and the URL of the current arena deck
lettuce = 1381
address = 'https://www.heartharena.com/arena-run/49xur1'

# Load the model, card dictionary
model = tf.keras.models.load_model(os.path.join('models', 'OModel5316.h5'))
card_dict = pickle.load(open(os.path.join('data', 'ODict5316.pkl'), 'rb'))

# Preparing dictionaries to convert data into integers and one-hot tensors
class_dict = {'Druid': 0, 'Hunter': 1, 'Mage': 2, 'Paladin': 3, 'Priest': 4, 'Rogue': 5, 'Shaman': 6,
              'Warlock': 7, 'Warrior': 8}
deck_type = {'Aggro-Control': 0, 'Attrition': 1, 'Classic Aggro': 2, 'Classic Control': 3, 'Mid-Range': 4, 'Tempo': 5}
expansion = {'Vanilla': 0, 'BRM': 1, 'WOG': 2, 'Kara': 3, 'MSG': 4, 'Ungoro': 5, 'KFT': 6, 'KnC': 7, 'Woods': 8,
             'Boomsday': 9, 'Rumble': 10, 'RoS': 11}

# Scrape deck information from the URL. The lxml library must be installed even though it is not explicitly imported.
source = requests.get(address)
soup = BeautifulSoup(source.text, 'lxml')
cards_in_deck = soup.find('ul', class_='deckList')

# Create a one-hot tensor for the deck class
hs_class = soup.title.text.split()[0]
class_tensor = [0] * 9
class_tensor[class_dict[hs_class]] = 1

# Create a one-hot tensor for the deck archetype
archetype = soup.find('header', class_='deck-archetype-name').find('span')
archetype = str(archetype)
archetype = archetype[6:]
archetype = archetype.split('<br/>')[0]
archetype_tensor = [0] * 6
archetype_tensor[deck_type[archetype]] = 1

# Create a one-hot tensor according to which Hearthstone expansion is in play
expansion_tensor = [0] * 12
expansion_tensor[expansion['RoS']] = 1

# Normalize deck score to a value between 0 and 1
score = soup.find('section', class_='arenaDeckTierscore').find('span').text
deck_score = (float(score) - 50) / 30

# List all the cards in the deck
deck_list = []
for card in cards_in_deck.find_all('span', class_='name'):
    deck_list.append(card.text)

# Quantity of Each Card in Deck
quantity_list = []
for card in cards_in_deck.find_all('span', class_='quantity'):
    quantity_list.append(int(card.text))

# Form the card tensor. This is not a one-hot. There are 1620 known cards in the game, so the tensor size was 1620.
# When a card was present in the deck, it's corresponding index was set to the number of copies of that card in the
# deck. All other indices were set to zero. This method was found to be just as effective as making 30 x 1620-sized
# one-hots.
card_tensor = [0] * len(card_dict)

i = 0
while i < len(deck_list):
    try:
        card_tensor[card_dict[deck_list[i]]] = quantity_list[i]
    except KeyError:
        print('One card not recognized.')
        pass
    i += 1

# Combine data into a single tensor for input into the NN
combined_tensor = class_tensor + archetype_tensor + expansion_tensor + [deck_score] + card_tensor
np_combined = np.array([combined_tensor])

# Run the tensor through the NN for prediction
prediction = model.predict(np_combined)

# Apply the Kelly criterion to determine the optimal wager amount
wager = (max(prediction[0]) - .5) * 2 * lettuce
reverse_wager = (.5 - min(prediction[0])) * 2 * lettuce

bracket_dict = {0: '0-4', 1: '5-8', 2: '9-12'}

print('Win Bracket Probabilities:')
print(f' 0-4: {prediction[0][0]:.02%}')
print(f' 5-8: {prediction[0][1]:.02%}')
print(f'9-12: {prediction[0][2]:.02%}')

if wager <= 0:
    print("No bet is at least 50% likely to win. They're all losers. Don't place a bet!")
    print(f"If possible, you'd want to bet {int(reverse_wager)} lettuce that he won't get "
          f"{bracket_dict[int(np.argmin(prediction[0]))]} wins.")
else:
    print('Bet', int(wager), 'lettuce.')
