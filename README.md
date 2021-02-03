# KrippTF
The Hearthstone player Kripparrian in his Twitch channel would allow the betting of an
in-channel currency dubbed "lettuce." Upon him completing the drafting of a deck while playing arena mode, he gave his viewers a minute or two to wager their lettuce that he would get either 0-4 wins, 5-8 wins, or
9-12 wins with that deck in that arena run. Winners would have their lettuce wagers doubled. Losers lost their
wager. This was a project aimed at maximizing lettuce while learning Tensorflow's Keras API.

wager_maker.py is the final product. Modules in the preliminaries folder were used to gather data and create the model that wager_maker uses.

# How It Works

Because Kripp used a certain deck-tracking add-on, his deck was immediately uploaded to a website upon completion of the deck draft. This module 1) web-scraped all deck information with Beautiful Soup, 2) processed it through a deep neural network which had been previously trained on Kripparrian's prior gameplay (about 1600 games), and 3) applied the Kelly criterion according to the DNN's confidence level, and ultimately returned an optimal waging strategy.

# Results & Analysis
I quickly jumped to the top 10% of currency owners before Kripp stopped playing arena (which occurred soon after making this).

An unexpected finding was that most bets were bad ones. For a bet to be placed, one of the win brackets needed to have a likelihood above 50%. No matter how good or bad a deck draft was, that was an infrequent occurrence. This observation likely gives a degree of quantitative corroboration to player opinion about arena at that time: that success in that game mode was leaning far too much on events outside of player control rather than intelligent deck drafting or skillful play. The most prominent of those events being the random 50/50 of which player got to go first. The player going first in arena mode had a sizable advantage.

# Improvements
If I were to develop this further, I would go to the trouble of [temperature scaling](https://arxiv.org/abs/1706.04599) for more accurate probabilities.
Kripp, however, doesn't play arena anymore. And as is, it worked well enough to quickly catapult me to the top 10% of
lettuce holders before he stopped playing. The over-confidence which is usually seen in neural network outputs did not seem to be present in the model that was used.
