import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import pandas as pd

state_dim = 36
action_dim = 2
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
BATCH_SIZE = 32

# Game Environment
class DiamantGame:
    def __init__(self):
        self.reset()

    def reset(self,nbr_joueurs = 3):
        # Initialize/reset the game state
        self.diamonds_collected = 0
        self.cards_played = []
        self.is_game_over = False
        self.deck = [
            {
                "type": "Trésor",
                "value": 3
            },
            {
                "type": "Trésor",
                "value": 4
            },
            {
                "type": "Trésor",
                "value": 14
            },
            {
                "type": "Trésor",
                "value": 13
            },
            {
                "type": "Trésor",
                "value": 2
            },
            {
                "type": "Trésor",
                "value": 15
            },
            {
                "type": "Trésor",
                "value": 17
            },
            {
                "type": "Trésor",
                "value": 7
            },
            {
                "type": "Trésor",
                "value": 9
            },
            {
                "type": "Trésor",
                "value": 1
            },
            {
                "type": "Trésor",
                "value": 5
            },
            {
                "type": "Trésor",
                "value": 11
            },
            {
                "type": "Trésor",
                "value": 11
            },
            {
                "type": "Trésor",
                "value": 5
            },
            {
                "type": "Trésor",
                "value": 7
            },
            {
                "type": "Relique",
                "value": 7
            },
            {
                "type": "Relique",
                "value": 5
            },
            {
                "type": "Relique",
                "value": 8
            },
            {
                "type": "Relique",
                "value": 10
            },
            {
                "type": "Relique",
                "value": 12
            },
            {
                "type": "Danger",
                "value": "Araignée"
            },
            {
                "type": "Danger",
                "value": "Araignée"
            },
            {
                "type": "Danger",
                "value": "Araignée"
            },
            {
                "type": "Danger",
                "value": "Lave"
            },
            {
                "type": "Danger",
                "value": "Lave"
            },
            {
                "type": "Danger",
                "value": "Lave"
            },
            {
                "type": "Danger",
                "value": "Pierre"
            },
            {
                "type": "Danger",
                "value": "Pierre"
            },
            {
                "type": "Danger",
                "value": "Pierre"
            },
            {
                "type": "Danger",
                "value": "Serpent"
            },
            {
                "type": "Danger",
                "value": "Serpent"
            },
            {
                "type": "Danger",
                "value": "Serpent"
            },
            {
                "type": "Danger",
                "value": "Pique"
            },
            {
                "type": "Danger",
                "value": "Pique"
            },
            {
                "type": "Danger",
                "value": "Pique"
            },
        ]
        self.nbr_joueurs = nbr_joueurs
        self.last_action = 0


    def draw_card(self):
        # Draw a random card from the deck
        card = random.choice(self.deck)
        self.deck.remove(card)
        return card
    
    def play_turn(self, action):

        reward = 0

        self.last_action = action

        if action == 1:
            self.is_game_over = True
            if self.diamonds_collected == 0:
                reward = -5
            elif self.diamonds_collected < 4:
                reward = -2
            else:
                reward = self.diamonds_collected * 2  # Reward for leaving safely
            return self.diamonds_collected, reward, self.is_game_over

        reward = 0
        card = self.draw_card()
        self.cards_played.append(card)

        card_value = card['value']
        card_type = card['type']

        if card_type == 'Trésor':
            self.diamonds_collected += card_value  # Reward for collecting diamonds
        elif card_type == 'Relique':
            self.diamonds_collected += 1
        elif card_type == 'Danger':
            dangerAlreadyExist = False
            for card in self.cards_played:
                if card['type'] == 'Danger':
                    if card['value'] == card_value:
                        dangerAlreadyExist = True
            if dangerAlreadyExist:
                self.is_game_over = True
                self.diamonds_collected = -5

        return self.diamonds_collected, reward, self.is_game_over

    def get_state(self):
        state = [0 if self.last_action == 0 else self.diamonds_collected]
        for card in self.cards_played:
            if card['type'] == 'Trésor':
                state.append(1)
            elif card['type'] == 'Relique':
                state.append(7)
            elif card['type'] == 'Danger':
                if card['value'] == 'Araignée':
                    state.append(2)
                elif card['value'] == 'Lave':
                    state.append(3)
                elif card['value'] == 'Pierre':
                    state.append(4)
                elif card['value'] == 'Serpent':
                    state.append(5)
                elif card['value'] == 'Pique':
                    state.append(6)

        for cards_not_played in range(35 - len(self.cards_played)):
            state.append(0)

        return state

# Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Reinforcement Learning Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []  # Memory for experience replay
        self.gamma = 0.99  # Discount factor

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice([0, 1])  # Randomly choose 'continue' or 'leave'
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0) # Debugging line
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = random.sample(self.memory, BATCH_SIZE)
        # Unpack transitions
        states, actions, next_states, rewards, dones = zip(*transitions)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute Q-values for current states
        Q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the expected Q-values for next states
        next_Q_values = self.model(next_states).max(1)[0]
        expected_Q_values = rewards + (self.gamma * next_Q_values * (1 - dones))

        # Loss
        loss = torch.nn.functional.mse_loss(Q_values, expected_Q_values.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_agent(episodes, state_dim, action_dim, epsilon_start, epsilon_end, epsilon_decay):

    game = DiamantGame()
    agent = DQNAgent(state_dim, action_dim)
    epsilon = epsilon_start

    rewards = []
    nbr_games = 0

    for episode in range(episodes):
        game.reset()
        current_state = game.get_state()
        total_reward = 0

        while not game.is_game_over:
            action = agent.select_action(current_state, epsilon)
            _, reward, done = game.play_turn(action)
            next_state = game.get_state()
            if action == 1:
                done = True
            agent.memory.append((current_state, action, next_state, reward, done))
            agent.optimize_model()

            current_state = next_state
            total_reward += next_state[0]

        rewards.append(total_reward)
        nbr_games += 1
        print(f"Episode: {episode + 1}, Reward: {total_reward}")

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

    return rewards, agent

def start():

    rewards, trained_agent = train_agent(100000, state_dim, action_dim, epsilon_start, epsilon_end, epsilon_decay)

    torch.save(trained_agent.model, 'diamant_model.pth')

    # make a percentage of the rewards and show the upgradings
    rewards = pd.Series(rewards)
    rewards = rewards.rolling(100, min_periods=1).mean()
    plt.plot(rewards)
    plt.show()



start()
