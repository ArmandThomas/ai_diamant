# Diamant its a game where you have to exit at good time from the cave.
# You have to collect diamonds and avoid dangers.
# Your actions is to continue, or exit and save your diamonds.
# If you exit you can't go back to the cave.
# If u have 2 same dangers in a row, you die.
# My game is aldreay finished, but i want to train a IA to play it.
# I want to train a ia with reinforcement learning to play it.

# In a row you have 35 cards : [{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e01"
#   },
#   "type": "Trésor",
#   "value": 3,
#   "url": "https://i.ibb.co/VMrM9KC/3-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e03"
#   },
#   "type": "Trésor",
#   "value": 14,
#   "url": "https://i.ibb.co/0DQDT9d/14-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e04"
#   },
#   "type": "Trésor",
#   "value": 5,
#   "url": "https://i.ibb.co/4SRy4KW/5-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e07"
#   },
#   "type": "Trésor",
#   "value": 2,
#   "url": "https://i.ibb.co/F7X3xCs/2-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e08"
#   },
#   "type": "Trésor",
#   "value": 15,
#   "url": "https://i.ibb.co/c38Tdxq/15-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e0b"
#   },
#   "type": "Trésor",
#   "value": 5,
#   "url": "https://i.ibb.co/4SRy4KW/5-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e10"
#   },
#   "type": "Danger",
#   "value": "Araignée",
#   "url": "https://i.ibb.co/NCgtsz9/araign-e.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e17"
#   },
#   "type": "Danger",
#   "value": "Serpent",
#   "url": "https://i.ibb.co/vvqnW3g/serpent.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e18"
#   },
#   "type": "Danger",
#   "value": "Serpent",
#   "url": "https://i.ibb.co/vvqnW3g/serpent.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e1f"
#   },
#   "type": "Relique",
#   "value": 8,
#   "url": "https://i.ibb.co/v3W4b8m/8-relique.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e05"
#   },
#   "type": "Trésor",
#   "value": 13,
#   "url": "https://i.ibb.co/34f228n/13-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e0d"
#   },
#   "type": "Trésor",
#   "value": 7,
#   "url": "https://i.ibb.co/dGxnf7j/7-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e0e"
#   },
#   "type": "Danger",
#   "value": "Araignée",
#   "url": "https://i.ibb.co/NCgtsz9/araign-e.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e16"
#   },
#   "type": "Danger",
#   "value": "Lave",
#   "url": "https://i.ibb.co/d2DWsP4/lave.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e19"
#   },
#   "type": "Danger",
#   "value": "Serpent",
#   "url": "https://i.ibb.co/vvqnW3g/serpent.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e1a"
#   },
#   "type": "Danger",
#   "value": "Pique",
#   "url": "https://i.ibb.co/kJvzr1R/pique.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e1d"
#   },
#   "type": "Relique",
#   "value": 5,
#   "url": "https://i.ibb.co/f8LjW7V/5-relique.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1dff"
#   },
#   "type": "Trésor",
#   "value": 4,
#   "url": "https://i.ibb.co/z20fvp6/4-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e0a"
#   },
#   "type": "Trésor",
#   "value": 11,
#   "url": "https://i.ibb.co/ypQRWkL/11-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e0c"
#   },
#   "type": "Trésor",
#   "value": 7,
#   "url": "https://i.ibb.co/dGxnf7j/7-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e11"
#   },
#   "type": "Danger",
#   "value": "Pierre",
#   "url": "https://i.ibb.co/WKQCq0p/pierre.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e12"
#   },
#   "type": "Danger",
#   "value": "Pierre",
#   "url": "https://i.ibb.co/WKQCq0p/pierre.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e1b"
#   },
#   "type": "Danger",
#   "value": "Pique",
#   "url": "https://i.ibb.co/kJvzr1R/pique.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e1c"
#   },
#   "type": "Danger",
#   "value": "Pique",
#   "url": "https://i.ibb.co/kJvzr1R/pique.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e1e"
#   },
#   "type": "Relique",
#   "value": 7,
#   "url": "https://i.ibb.co/1vs54L5/7-relique.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e20"
#   },
#   "type": "Relique",
#   "value": 10,
#   "url": "https://i.ibb.co/ZTF23Vf/10-relique.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e00"
#   },
#   "type": "Trésor",
#   "value": 1,
#   "url": "https://i.ibb.co/QvznK0L/1-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e02"
#   },
#   "type": "Trésor",
#   "value": 9,
#   "url": "https://i.ibb.co/QXLfMfj/9-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e06"
#   },
#   "type": "Trésor",
#   "value": 11,
#   "url": "https://i.ibb.co/ypQRWkL/11-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e09"
#   },
#   "type": "Trésor",
#   "value": 17,
#   "url": "https://i.ibb.co/Drdj2mF/17-rubis.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e0f"
#   },
#   "type": "Danger",
#   "value": "Araignée",
#   "url": "https://i.ibb.co/NCgtsz9/araign-e.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e13"
#   },
#   "type": "Danger",
#   "value": "Pierre",
#   "url": "https://i.ibb.co/WKQCq0p/pierre.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e14"
#   },
#   "type": "Danger",
#   "value": "Lave",
#   "url": "https://i.ibb.co/d2DWsP4/lave.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e15"
#   },
#   "type": "Danger",
#   "value": "Lave",
#   "url": "https://i.ibb.co/d2DWsP4/lave.webp"
# },{
#   "_id": {
#     "$oid": "642bd86e8bad57c6e4bd1e21"
#   },
#   "type": "Relique",
#   "value": 12,
#   "url": "https://i.ibb.co/kKk892P/12-relique.webp"
# }]
import random


class Diamant:
    def __init__(self):
        self.deck = [
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e01"
              },
              "type": "Trésor",
              "value": 3,
              "url": "https://i.ibb.co/VMrM9KC/3-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e03"
              },
              "type": "Trésor",
              "value": 14,
              "url": "https://i.ibb.co/0DQDT9d/14-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e04"
              },
              "type": "Trésor",
              "value": 5,
              "url": "https://i.ibb.co/4SRy4KW/5-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e07"
              },
              "type": "Trésor",
              "value": 2,
              "url": "https://i.ibb.co/F7X3xCs/2-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e08"
              },
              "type": "Trésor",
              "value": 15,
              "url": "https://i.ibb.co/c38Tdxq/15-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e0b"
              },
              "type": "Trésor",
              "value": 5,
              "url": "https://i.ibb.co/4SRy4KW/5-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e10"
              },
              "type": "Danger",
              "value": "Araignée",
              "url": "https://i.ibb.co/NCgtsz9/araign-e.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e17"
              },
              "type": "Danger",
              "value": "Serpent",
              "url": "https://i.ibb.co/vvqnW3g/serpent.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e18"
              },
              "type": "Danger",
              "value": "Serpent",
              "url": "https://i.ibb.co/vvqnW3g/serpent.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e05"
              },
              "type": "Trésor",
              "value": 13,
              "url": "https://i.ibb.co/34f228n/13-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e0d"
              },
              "type": "Trésor",
              "value": 7,
              "url": "https://i.ibb.co/dGxnf7j/7-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e0e"
              },
              "type": "Danger",
              "value": "Araignée",
              "url": "https://i.ibb.co/NCgtsz9/araign-e.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e16"
              },
              "type": "Danger",
              "value": "Lave",
              "url": "https://i.ibb.co/d2DWsP4/lave.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e19"
              },
              "type": "Danger",
              "value": "Serpent",
              "url": "https://i.ibb.co/vvqnW3g/serpent.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e1a"
              },
              "type": "Danger",
              "value": "Pique",
              "url": "https://i.ibb.co/kJvzr1R/pique.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1dff"
              },
              "type": "Trésor",
              "value": 4,
              "url": "https://i.ibb.co/z20fvp6/4-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e0a"
              },
              "type": "Trésor",
              "value": 11,
              "url": "https://i.ibb.co/ypQRWkL/11-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e0c"
              },
              "type": "Trésor",
              "value": 7,
              "url": "https://i.ibb.co/dGxnf7j/7-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e11"
              },
              "type": "Danger",
              "value": "Pierre",
              "url": "https://i.ibb.co/WKQCq0p/pierre.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e12"
              },
              "type": "Danger",
              "value": "Pierre",
              "url": "https://i.ibb.co/WKQCq0p/pierre.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e1b"
              },
              "type": "Danger",
              "value": "Pique",
              "url": "https://i.ibb.co/kJvzr1R/pique.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e1c"
              },
              "type": "Danger",
              "value": "Pique",
              "url": "https://i.ibb.co/kJvzr1R/pique.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e00"
              },
              "type": "Trésor",
              "value": 1,
              "url": "https://i.ibb.co/QvznK0L/1-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e02"
              },
              "type": "Trésor",
              "value": 9,
              "url": "https://i.ibb.co/QXLfMfj/9-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e06"
              },
              "type": "Trésor",
              "value": 11,
              "url": "https://i.ibb.co/ypQRWkL/11-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e09"
              },
              "type": "Trésor",
              "value": 17,
              "url": "https://i.ibb.co/Drdj2mF/17-rubis.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e0f"
              },
              "type": "Danger",
              "value": "Araignée",
              "url": "https://i.ibb.co/NCgtsz9/araign-e.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e13"
              },
              "type": "Danger",
              "value": "Pierre",
              "url": "https://i.ibb.co/WKQCq0p/pierre.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e14"
              },
              "type": "Danger",
              "value": "Lave",
              "url": "https://i.ibb.co/d2DWsP4/lave.webp"
            },
            {
              "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e15"
              },
              "type": "Danger",
              "value": "Lave",
              "url": "https://i.ibb.co/d2DWsP4/lave.webp"
            }]
        self.state_deck = []
        self.balance = 0

    def user_action(self, action):
        if action == 'continue':
            its_ok = self.draw_card()
            if its_ok == False:
                return False, 0
        elif action == 'stop':
            return False, self.balance

    def draw_card(self):
        card = random.choice(self.deck)
        self.deck.remove(card)
        self.state_deck.append(card)
        if card['type'] == 'Trésor':
            self.balance += card['value']
        elif card['type'] == 'Danger':
            # find in state_deck contain the same value
            if len([card for card in self.state_deck if card['value'] == card['value']]) == 2:
                self.balance = 0
                return False
        return True

# Now want to train the agent to play the game
# We will use a Q-learning algorithm

# First we need to define the state space
# The state space is the set of all possible states the agent can be in
# In this case, the state space is the set of all possible cards in the deck
# We will represent the state space as a list of cards
# Each card will be represented as a dictionary with the following keys:
#   - type: the type of card (Trésor or Danger)
#   - value: the value of the card
#   - url: the url of the card image

# We will also need to define the action space
# The action space is the set of all possible actions the agent can take
# In this case, the action space is the set of all possible actions the user can take
# We will represent the action space as a list of actions
# Each action will be represented as a string
# The possible actions are:
#   - continue: the user continues to draw cards
#   - stop: the user stops drawing cards

# We will also need to define the reward function
# The reward function is a function that takes in a state and an action and returns a reward
# In this case, the reward function is a function that takes in a state and an action and returns the balance
# The balance is the amount of money the user has won
# The balance is initialized to 0
# The balance is increased by the value of the card if the card is a Trésor
# The balance is returned to 0 if user draws two cards danger with the same value

# We will also need to define the transition function
# The transition function is a function that takes in a state and an action and returns a new state
# In this case, the transition function is a function that takes in a state and an action and returns a new state
# The new state is the state after the user has taken the action
# The new state is the state after the user has drawn a card
# Here is the transition function:
#   - if the action is continue, draw a card
#   - if the action is stop, return the balance

# We will also need to define the Q-function
# The Q-function is a function that takes in a state and an action and returns a Q-value
# The Q-value is the expected reward of taking the action in the state
# The Q-value is the sum of the reward of taking the action in the state and the Q-value of the next state
# The Q-value of the next state is the maximum Q-value of the next state

import random
import numpy as np


class Diamant:
    def __init__(self):
        self.original_deck = [{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e01"
            },
            "type": "Trésor",
            "value": 3,
            "url": "https://i.ibb.co/VMrM9KC/3-rubis.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e03"
            },
            "type": "Trésor",
            "value": 14,
            "url": "https://i.ibb.co/0DQDT9d/14-rubis.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e04"
            },
            "type": "Trésor",
            "value": 5,
            "url": "https://i.ibb.co/4SRy4KW/5-rubis.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e07"
            },
            "type": "Trésor",
            "value": 2,
            "url": "https://i.ibb.co/F7X3xCs/2-rubis.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e08"
            },
            "type": "Trésor",
            "value": 15,
            "url": "https://i.ibb.co/c38Tdxq/15-rubis.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e0b"
            },
            "type": "Trésor",
            "value": 5,
            "url": "https://i.ibb.co/4SRy4KW/5-rubis.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e10"
            },
            "type": "Danger",
            "value": "Araignée",
            "url": "https://i.ibb.co/NCgtsz9/araign-e.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e17"
            },
            "type": "Danger",
            "value": "Serpent",
            "url": "https://i.ibb.co/vvqnW3g/serpent.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e18"
            },
            "type": "Danger",
            "value": "Serpent",
            "url": "https://i.ibb.co/vvqnW3g/serpent.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e05"
            },
            "type": "Trésor",
            "value": 13,
            "url": "https://i.ibb.co/34f228n/13-rubis.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e0d"
            },
            "type": "Trésor",
            "value": 7,
            "url": "https://i.ibb.co/dGxnf7j/7-rubis.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e0e"
            },
            "type": "Danger",
            "value": "Araignée",
            "url": "https://i.ibb.co/NCgtsz9/araign-e.webp"
        },{
            "_id": {
                "$oid": "642bd86e8bad57c6e4bd1e16"
            },
            "type": "Danger",
            "value": "Lave",
            "url": "https://i.ibb.co/d2DWsP4/lave.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e19"
          },
          "type": "Danger",
          "value": "Serpent",
          "url": "https://i.ibb.co/vvqnW3g/serpent.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e1a"
          },
          "type": "Danger",
          "value": "Pique",
          "url": "https://i.ibb.co/kJvzr1R/pique.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1dff"
          },
          "type": "Trésor",
          "value": 4,
          "url": "https://i.ibb.co/z20fvp6/4-rubis.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e0a"
          },
          "type": "Trésor",
          "value": 11,
          "url": "https://i.ibb.co/ypQRWkL/11-rubis.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e0c"
          },
          "type": "Trésor",
          "value": 7,
          "url": "https://i.ibb.co/dGxnf7j/7-rubis.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e11"
          },
          "type": "Danger",
          "value": "Pierre",
          "url": "https://i.ibb.co/WKQCq0p/pierre.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e12"
          },
          "type": "Danger",
          "value": "Pierre",
          "url": "https://i.ibb.co/WKQCq0p/pierre.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e1b"
          },
          "type": "Danger",
          "value": "Pique",
          "url": "https://i.ibb.co/kJvzr1R/pique.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e1c"
          },
          "type": "Danger",
          "value": "Pique",
          "url": "https://i.ibb.co/kJvzr1R/pique.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e00"
          },
          "type": "Trésor",
          "value": 1,
          "url": "https://i.ibb.co/QvznK0L/1-rubis.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e02"
          },
          "type": "Trésor",
          "value": 9,
          "url": "https://i.ibb.co/QXLfMfj/9-rubis.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e06"
          },
          "type": "Trésor",
          "value": 11,
          "url": "https://i.ibb.co/ypQRWkL/11-rubis.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e09"
          },
          "type": "Trésor",
          "value": 17,
          "url": "https://i.ibb.co/Drdj2mF/17-rubis.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e0f"
          },
          "type": "Danger",
          "value": "Araignée",
          "url": "https://i.ibb.co/NCgtsz9/araign-e.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e13"
          },
          "type": "Danger",
          "value": "Pierre",
          "url": "https://i.ibb.co/WKQCq0p/pierre.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e14"
          },
          "type": "Danger",
          "value": "Lave",
          "url": "https://i.ibb.co/d2DWsP4/lave.webp"
        },{
          "_id": {
            "$oid": "642bd86e8bad57c6e4bd1e15"
          },
          "type": "Danger",
          "value": "Lave",
          "url": "https://i.ibb.co/d2DWsP4/lave.webp"
        }
    ] # Replace with your original cards list
        self.reset()

    def reset(self):
        self.deck = self.original_deck.copy()
        random.shuffle(self.deck)
        self.state_deck = []
        self.balance = 0
        self.game_over = False
        return self.get_current_state()

    def step(self, action):
        if action == 'continue':
            self.game_over, reward = self.draw_card()
        elif action == 'stop':
            self.game_over, reward = True, self.balance
        return self.get_current_state(), reward, self.game_over

    def draw_card(self):
        if not self.deck:
            return True, self.balance  # End game if no cards left

        card = self.deck.pop()
        self.state_deck.append(card)

        if card['type'] == 'Trésor':
            self.balance += card['value']
            return False, 0  # No immediate reward for continuing
        elif card['type'] == 'Danger':
            if len([c for c in self.state_deck if c['value'] == card['value']]) == 2:
                self.balance = 0
                return True, 0  # Game ends with no reward
            return False, 0

    def get_current_state(self):
        # Convert current state_deck to a state representation
        state = {'Trésor': 1, 'Danger': [
            {'Pierre': 2, 'Pique': 3, 'Araignée': 4, 'Lave': 5, 'Serpent': 6}
        ]}
        for card in self.state_deck:
            state[card['type']] += 1
        return state


class QLearningAgent:
    def __init__(self, alpha, gamma, n_actions, epsilon):
        self.Q = np.zeros((n_actions, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = ['continue', 'stop']
        self.n_actions = n_actions

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def learn(self, state, action, reward, next_state):
        action_index = self.actions.index(action)
        predict = self.Q[state, action_index]
        target = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action_index] += self.alpha * (target - predict)

def train_model(game, agent, num_episodes):
    for episode in range(num_episodes):
        state = game.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = game.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

# Initialize game and agent
game = Diamant()
agent = QLearningAgent(alpha=0.1, gamma=0.9, n_actions=2, epsilon=0.1, deck_size=len(game.original_deck))
