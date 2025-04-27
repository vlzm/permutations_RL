import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512]):
        """
        Deep Q-Network for permutation learning
        
        Args:
            input_size (int): Size of input features
            output_size (int): Size of output (number of actions)
            hidden_sizes (list): List of hidden layer sizes
        """
        super().__init__()
        
        # Store sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        # Create layers
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
            
        # Add output layer with sigmoid activation
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Handle input type
        if x.dtype != torch.float32:
            x = x.float()
            
        # Convert to one-hot if needed
        if len(x.shape) == 2 and x.shape[1] != self.input_size:
            batch_size = x.shape[0]
            n = int(np.sqrt(self.input_size))
            one_hot = torch.zeros(batch_size, n, n, device=x.device)
            
            # Convert indices to long type and ensure they're on the correct device
            row_idx = torch.arange(n, device=x.device).long()
            for i in range(batch_size):
                col_idx = x[i].long()  # Convert to long type
                one_hot[i, row_idx, col_idx] = 1
                
            x = one_hot.view(batch_size, -1)
            
        return self.model(x)
    
    def to(self, device):
        """Override to() to ensure all components move to the same device"""
        super().to(device)
        self.model = self.model.to(device)
        return self

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Experience replay buffer
        
        Args:
            capacity (int): Maximum size of buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample random batch from buffer
        
        Args:
            batch_size (int): Size of batch to sample
            
        Returns:
            tuple: Batch of experiences
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        """Get current size of buffer"""
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, config):
        """
        DQN Agent
        
        Args:
            state_size (int): Size of state representation
            action_size (int): Number of possible actions
            config (dict): Configuration dictionary
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Create networks
        self.policy_net = DQN(state_size, action_size, config.get('hidden_sizes', [512]))
        self.target_net = DQN(state_size, action_size, config.get('hidden_sizes', [512]))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )
        
        # Replay buffer
        self.memory = ReplayBuffer(config.get('buffer_size', 10000))
        
    def act(self, state):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
        
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.push(state, action, reward, next_state, done)
        
    def replay(self):
        """
        Train on batch of experiences
        """
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        """Update target network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict()) 