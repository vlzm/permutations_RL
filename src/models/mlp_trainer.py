import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from tqdm import tqdm

class MLPTrainer:
    def __init__(self, model, config):
        """
        Initialize MLP trainer
        
        Args:
            model: PyTorch model to train
            config: Dictionary containing training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.batch_size = config.get('batch_size', 1024)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.n_random_walks_to_generate = config.get('n_random_walks_to_generate', 10000)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.criterion = nn.MSELoss().to(self.device)
        
        # Training state
        self.epoch_losses = []
        
    def train_epoch(self, X, y):
        
        # Shuffle train data and 
        y_train = y.float().to(self.device)
        indices = torch.randperm(X.shape[0], dtype=torch.int64, device='cpu')  # Generate indices on CPU
        X_train = X[indices].to(self.device)
        y_train = y_train[indices]
        
        self.model.train()
        # Neural network train by batches (not to crash RAM)
        n_states_all =  X_train.shape[0]
        cc = 0; train_loss = 0.0
        for i_start_batch  in range(0,n_states_all,self.batch_size ):
            i_end_batch = min( [i_start_batch + self.batch_size,  n_states_all ] )
            
            # Forward
            outputs = self.model(X_train[i_start_batch:i_end_batch])
            loss = self.criterion(outputs.squeeze(), y_train[i_start_batch:i_end_batch])
            
            # Backward and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item(); 
            cc+=1
        train_loss /= cc
        
        return train_loss
        
    def evaluate(self, X, y):
        """Evaluate model on test data"""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)
            outputs = self.model(X)
            outputs = outputs.view(-1)
            loss = self.criterion(outputs, y.float())
        return loss.item()
        
    def get_loss_history(self):
        """Get training loss history"""
        return self.epoch_losses
        
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.epoch_losses,
        }, path)
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch_losses = checkpoint['loss_history'] 