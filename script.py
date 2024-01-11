import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Game Environment
class DiamantGame:
    def __init__(self):
        self.reset()

    def reset(self):
        # Initialize/reset the game state
        self.diamonds_collected = 0
        self.danger_cards_drawn = []
        self.is_game_over = False
        self.deck = self.create_deck()

    def create_deck(self):
        # Create a deck of cards (15 diamond cards, 15 danger cards)
        diamond_cards = [(diamonds, 'Trésor') for diamonds in range(1, 16)]
        danger_types = ['Araignée', 'Pierre', 'Lave', 'Serpent', 'Pique']
        danger_cards = [(danger, 'Danger') for danger in danger_types for _ in range(3)]
        deck = diamond_cards + danger_cards
        np.random.shuffle(deck)
        return deck

    def draw_card(self):
        # Draw a card from the deck
        if not self.deck:
            return None
        return self.deck.pop()

    def play_turn(self, action):
        # Updated method to include reward calculation
        reward = 0
        if action == 1:
            self.is_game_over = True
            reward = self.diamonds_collected  # Reward for leaving safely
            return self.diamonds_collected, reward, self.is_game_over

        card = self.draw_card()

        card_value, card_type = card
        if card_type == 'Trésor':
            self.diamonds_collected += card_value # Reward for collecting diamonds
        elif card_type == 'Danger':
            if card_value in self.danger_cards_drawn:
                self.is_game_over = True
                self.diamonds_collected = 0
            else:
                self.danger_cards_drawn.append(card_value)

        return self.diamonds_collected, reward, self.is_game_over
    
    def get_state(self):
        # Convert the game state into a numerical vector
        state = [self.diamonds_collected] + [int(danger in self.danger_cards_drawn) for danger in ['Araignée', 'Pierre', 'Lave', 'Serpent', 'Pique']]
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


# Training Loop
def train_agent(episodes, state_dim, action_dim, epsilon_start, epsilon_end, epsilon_decay):
    game = DiamantGame()
    agent = DQNAgent(state_dim, action_dim)
    epsilon = epsilon_start

    rewards = []
    mean_rewards = []

    for episode in range(episodes):
        game.reset()
        current_state = game.get_state()
        total_reward = 0

        while not game.is_game_over:
            action = agent.select_action(current_state, epsilon)
            _, reward, done = game.play_turn(action)
            next_state = game.get_state()
            agent.memory.append((current_state, action, next_state, reward, done))
            agent.optimize_model()

            current_state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            mean_reward = np.mean(rewards[-100:])
            mean_rewards.append(mean_reward)
            print(f"Episodes {episode-99}-{episode}: Mean Reward: {mean_reward}")

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

    return rewards, mean_rewards


# Hyperparameters
state_dim = 6
action_dim = 2
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
BATCH_SIZE = 32

# Start Training
rewards, mean_rewards = train_agent(100000, state_dim, action_dim, epsilon_start, epsilon_end, epsilon_decay)

# Plotting
plt.plot(range(100, 100001, 100), mean_rewards)
plt.xlabel('Episodes')
plt.ylabel('Mean Reward')
plt.title('Mean Reward every 100 Episodes')
plt.show()