import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from math import exp

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=7, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=4 * 4 * 128, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=4)

    def load_model_weights(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print('Weights loaded')

    def forward(self, x, batch_size=1):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(batch_size, -1) if batch_size > 1 else x.view(-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class DQN_agent:
    def __init__(self, epsilon=0.5):
        self.total_steps = 0
        self.model = DQN()
        self.lr = 0.001
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.state_input = [torch.zeros((64, 64, 3)) for _ in range(4)]  # Initialize with zeros
        self.replay_buffer1 = deque(maxlen=10000)
        self.replay_buffer2 = deque(maxlen=10000)
        self.eta = 0.8
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def select_action(self, game_state):
        self.state_input.pop(0)
        self.state_input.append(torch.tensor(game_state))
        self.total_steps += 1

        decay_factor = 0.99  # Adjust this value based on your desired decay rate

        if random.random() < max(self.epsilon * exp(-self.total_steps / 100000 * decay_factor), 0.05):
            # Your logic here

            action = random.randint(1, 4)
        else:
            state_input_tensor = torch.cat(self.state_input, dim=2).to(dtype=torch.float32).permute(2, 0, 1)
            q_values = self.model(state_input_tensor)  # todo : self.target_model
            action = torch.argmax(q_values).item() + 1
        return action

    def add_experience(self, state, action, reward, next_state, game_over):
        state_tensor = torch.cat(state, dim=2).to(dtype=torch.float32).permute(2, 0, 1)
        next_state_tensor = torch.cat(next_state, dim=2).to(dtype=torch.float32).permute(2, 0, 1)
        experience = (state_tensor, action, reward, next_state_tensor, game_over)
        if reward >= 10:
            self.replay_buffer1.append(experience)
        else:
            self.replay_buffer2.append(experience)

    def update_q_network(self, batch_size, gamma):
        if len(self.replay_buffer1) + len(self.replay_buffer2) < batch_size:
            return
        combined_buffer = self.replay_buffer1 + self.replay_buffer2
        batch = random.sample(combined_buffer, batch_size) if len(combined_buffer) > batch_size else combined_buffer
        states = torch.stack([experience[0] for experience in batch]).to(dtype=torch.float32)
        actions = torch.tensor([experience[1] for experience in batch], dtype=torch.long)
        rewards = torch.tensor([experience[2] for experience in batch], dtype=torch.float32)
        next_states = torch.stack([experience[3] for experience in batch]).to(dtype=torch.float32)
        dones = torch.tensor([experience[4] for experience in batch], dtype=torch.bool)

        # Compute Q-values for current and next states
        q_values = self.model(states, batch_size=len(batch))
        next_q_values = self.target_model(next_states, batch_size=len(batch))

        # Calculate the target Q-values
        target_q_values = q_values.clone()
        target = rewards + gamma * torch.max(next_q_values, dim=1).values * ~dones
        target_q_values[torch.arange(len(batch)), actions - 1] = target

        # Calculate the DQN loss
        loss = nn.MSELoss()(q_values, target_q_values.detach())

        # Update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self, update_frequency):
        if self.total_steps % update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

