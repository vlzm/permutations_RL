import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
import copy
from tqdm import tqdm
from src.utils.random_walks import random_walks_nbt, get_neighbors
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import json
import os
class DQNTrainer:
    def __init__(self,
                 model,
                 X_anchor,
                 y_anchor,
                 criterion,
                 optimizer,
                 list_generators,
                 tensor_generators,
                 cfg,
                 state_destination,
                 random_walks_type,
                 device):
        """
        Initializes the DQN trainer.
        Args:
            model: PyTorch model.
            criterion: Loss function.
            optimizer: Optimizer.
            list_generators: List of data generators.
            tensor_generators: Tensor of generators for neighbor lookup.
            cfg: Configuration dict with keys:
                - n_epochs_dqn
                - flag_dqn_round
                - n_random_walks_to_generate_dqn
                - n_random_walk_length
                - n_random_walks_steps_back_to_ban
                - dqn_batch_size
            state_destination: Initial state for random walks.
            random_walks_type: Type of random walks.
            device: Torch device.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.list_generators = list_generators
        self.tensor_generators = tensor_generators
        self.cfg = cfg
        self.state_destination = state_destination
        self.random_walks_type = random_walks_type
        self.device = device
        self.w_anchor = cfg.get('w_anchor', 1.0)
        self.w_hinge = cfg.get('w_hinge', 1.0)
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()
        self.sync_freq = cfg.get('sync_freq', 100)
        self.w_tight = cfg.get('w_tight', 0.0)
        self.X_anchor = X_anchor
        self.y_anchor = y_anchor

    def _generate_data(self):
        """
        Generates random walk data and true distances.
        Returns:
            X_train: Tensor of states.
            y_train: Tensor of true distances.
        """
        return random_walks_nbt(self.state_destination,
            self.list_generators,
            n_random_walk_length=self.cfg['n_random_walk_length'],
            n_random_walks_to_generate=self.cfg['n_random_walks_to_generate_dqn'],
            n_random_walks_steps_back_to_ban=self.cfg['n_random_walks_steps_back_to_ban'],
            random_walks_type=self.random_walks_type,
            state_rw_start=self.state_destination
        )

    def _compute_neighbors(self, X):
        """
        Computes neighbor tensors for a batch of states.
        """
        return get_neighbors(X, self.tensor_generators)
    
    def get_unique_log_dir(self, base_name="runs/experiment_dqn"):
        i = 1
        log_dir = base_name
        while os.path.exists(log_dir):
            log_dir = f"{base_name}_{i}"
            i += 1
        return log_dir
    
    
    #########################################################
    #########################################################
    #########################################################

    def _compute_bellman_targets(self, X_train, y_train):
        """
        Computes Bellman update targets for DQN.
        """
        n_states = X_train.shape[0]
        y_bellman = torch.zeros(n_states, device=self.device, dtype=torch.float)
        self.model.eval()
        with torch.no_grad():
            neigb = self._compute_neighbors(X_train)
            for start in range(0, n_states, self.cfg['dqn_batch_size']):
                end = min(start + self.cfg['dqn_batch_size'], n_states)
                y_pred = self.model(neigb[start:end])
                y_pred = 1 + torch.min(y_pred, dim=1)[0]
                y_bellman[start:end] = y_pred.reshape(-1)

        # Apply boundary conditions and clipping
        y_train = torch.min(y_bellman, y_train)
        y_train = torch.clamp_min(y_train, 1)
        y_train[:self.cfg['n_random_walks_to_generate_dqn']] = 0
        if self.cfg['flag_dqn_round']:
            y_train = torch.round(y_train)

        return y_train

    def _shuffle(self, X, y):
        """
        Shuffles training data.
        """
        perm = torch.randperm(X.shape[0])
        return X[perm], y[perm]

    def train_single_hard_hinge(self):
        """
        Main training loop for DQN.
        """
        self.dqn_exp_name = self.get_unique_log_dir()
        print('Training DQN with random walks')
        writer = SummaryWriter(f'{self.dqn_exp_name}')
        writer.add_text("config", json.dumps(self.cfg, indent=2))

        n_epochs = self.cfg['n_epochs_dqn']
        verbose = self.cfg.get('verbose_loc', 10)

        history = {'train_loss': []}
        start_time = time.time()
        print(f"Starting DQN training for {n_epochs} epochs...")
        pbar = tqdm(range(n_epochs), desc="Training MLP with random walks")

        for epoch in pbar:

            t0 = time.time()
            X_train, y_train = self._generate_data()
            if verbose >= 100:
                print(f"Epoch {epoch}: Generated data in {time.time() - t0:.2f}s")
            t_rw = time.time() - t0

            t0 = time.time()
            y_train = self._compute_bellman_targets(X_train, y_train)
            t_bellman = time.time() - t0

            X_train, y_train = self._shuffle(X_train, y_train)

            t0 = time.time()

            self.model.train()
            total_loss = 0.0
            count = 0
            n_states = X_train.shape[0]
            for start in range(0, n_states, self.cfg['dqn_batch_size']):
                end = min(start + self.cfg['dqn_batch_size'], n_states)
                batch_X = X_train[start:end]
                batch_y = y_train[start:end]

                outputs = self.model(batch_X)
                loss_hinge = self.criterion(outputs.squeeze(), batch_y)

                anchor = 0.0
                if self.w_anchor > 0:
                    idx = torch.randint(0, len(self.X_anchor), (batch_X.size(0),), device=self.device)
                    x_anchor = self.X_anchor[idx].to(self.device)
                    with torch.no_grad():
                        # таргет можно держать без градиента
                        target_anchor = self.y_anchor[idx].float().to(self.device)

                    h_anchor = self.model(x_anchor).squeeze()       # ← есть градиент
                    anchor = F.mse_loss(h_anchor, target_anchor, reduction='mean')

                loss = self.w_hinge * loss_hinge + self.w_anchor * anchor

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

            train_loss = total_loss / count

            history['train_loss'].append(train_loss)
            t_train = time.time() - t0

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/hinge', loss_hinge, epoch)
            writer.add_scalar('Loss/anchor', anchor, epoch)

            for name, param in self.model.named_parameters():
                writer.add_histogram(name, param.data, epoch)

            if epoch % verbose == 0:
                pbar.set_postfix(loss=f"{train_loss:.4f}, hinge: {loss_hinge:.4f}, anchor: {anchor:.4f}")

        print(f"Training finished in {time.time() - start_time:.1f}s")
        writer.close()
        return history