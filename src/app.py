import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
from src.models.mlp import PermutationMLP, PermutationQuadMLP, PermutationQuadEquivariantMLP, DeepSetMLP, EquivariantSumMLP
from src.models.dqn import DQN, DQNAgent, ReplayBuffer
from src.models.mlp_trainer import MLPTrainer, QuadMLPTrainer
from src.models.dqn_trainer import DQNTrainer
from src.search.beam_search import beam_search_torch, initialize_states, beam_search_path
from src.utils.random_walks import get_neighbors
from src.utils.random_walks import random_walks_nbt
from src.utils.anchor import bfs_build_dataset, build_quadruples_from_bfs
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import json
import os
from src.utils.device_manager import DeviceManager

class PermutationSolver:
    def __init__(self, config=None, gpu_id=None):
        if config is None:
            config = self.get_default_config()
        self.config = config
        
        # Initialize device manager
        self.device_manager = DeviceManager(gpu_id)
        self.device = self.device_manager.get_device()
        
        # Models and trainers
        self.mlp_model = None
        self.dqn_model = None
        self.mlp_trainer = None
        self.dqn_trainer = None
        
        # Random-walk components (initialized later)
        self.list_generators = None
        self.tensor_generators = None
        self.state_destination = None
        
        # Mode
        self.mode = config.get('mode', 'single_hard_hinge')

        # Training data
        self.X_anchor = None
        self.y_anchor = None
        self.X_test = None
        self.y_test = None
        
        self.mlp_exp_name = None
        self.dqn_exp_name = None
    
    def get_unique_log_dir(self, base_name="runs/experiment_mlp"):
        i = 1
        log_dir = base_name
        while os.path.exists(log_dir):
            log_dir = f"{base_name}_{i}"
            i += 1
        return log_dir
    
    def get_default_config(self):
        n = 12  # n_permutations_length
        return {
            'n_permutations_length': n,
            
            # Random walks params
            'random_walks_type': 'non-backtracking-beam',
            'n_random_walk_length': int(n * (n-1) / 2),
            'n_random_walks_to_generate': 10000,
            'n_random_walks_steps_back_to_ban': 8,
            
            # Neural Net params
            'model_type': 'MLP',
            'list_layers_sizes': [2**9],
            'n_epochs': 150,
            'mlp_batch_size': 1024,
            'dqn_batch_size': 1024,
            'lr_supervised': 0.0001,
            
            # DQN training
            'n_epochs_dqn': 300,
            'flag_dqn_round': False,
            'n_random_walks_to_generate_dqn': 1000,
            'lr_rl': 0.0001,
            
            # Beam search
            'beam_search_torch': True,
            'beam_search_Fironov': False,
            'beam_width': 2**16,
            'n_steps_limit': 4 * n**2,
            'alpha_previous_cost_accumulation': 0,
            'beam_search_models_or_heuristics': 'model_torch',
            'ban_p0_p1_transposition_if_p0_lt_p1_ie_already_sorted': False,
            'n_beam_search_steps_back_to_ban': 32,
            
            # What to solve
            'solve_random_or_longest_state': 'solve_LRX_longest',
            'verbose': 100
        }
    
    def _init_rw_generators(self):
        """
        Initialize random-walk generators and state.
        """
        n = self.config['n_permutations_length']
        L = np.concatenate([np.arange(1, n), [0]])
        R = np.concatenate([[n-1], np.arange(n-1)])
        X = np.concatenate([[1,0], np.arange(2, n)])
        self.list_generators = [L, R, X]
        # Precompute neighbors tensor if needed by trainer
        # Here, tensor_generators is simply passed along to DQNTrainer
        self.tensor_generators = torch.tensor(np.stack(self.list_generators), dtype=torch.int64, device=self.device)
        self.state_destination = torch.arange(n, device=self.device, dtype=torch.int64)
    
    def setup_mlp_model(self, mode='single'):
        """Initialize MLP model and trainer"""
        n = self.config['n_permutations_length']

        if self.list_generators is None:
            self._init_rw_generators()

        self.X_anchor, self.y_anchor, self.X_test, self.y_test = self.generate_training_data_anchor(mode=mode)

        if mode == 'quadruples':
            self.mlp_model = PermutationQuadEquivariantMLP(
                n=n,
                hidden_dims=self.config['list_layers_sizes'],
                num_classes=n
            ).to(self.device)

            self.mlp_trainer = QuadMLPTrainer(self.mlp_model, {
                'batch_size': self.config['mlp_batch_size'],
                'learning_rate': self.config['lr_supervised'],
                'hidden_sizes': self.config['list_layers_sizes'],
                'n_random_walks_to_generate': self.config['n_random_walks_to_generate']
            }, self.tensor_generators)

        elif mode == 'single':
            # self.mlp_model = EquivariantSumMLP(
            #     n=n,
            #     num_classes=n,
            #     phi_dim=self.config['list_layers_sizes'][0],
            #     mlp_hidden=self.config['list_layers_sizes'][1:]
            # ).to(self.device)

            self.mlp_model = PermutationMLP(
                input_size=n,
                hidden_dims=self.config['list_layers_sizes'],
                num_classes_for_one_hot=n
            ).to(self.device)

            self.mlp_trainer = MLPTrainer(self.mlp_model, {
            'batch_size': self.config['mlp_batch_size'],
            'learning_rate': self.config['lr_supervised'],
            'hidden_sizes': self.config['list_layers_sizes'],
            'n_random_walks_to_generate': self.config['n_random_walks_to_generate']
        })

             
    def setup_dqn_model(self):
        if self.mlp_model is None:
            raise ValueError("MLP model must be trained first")
        # Initialize random-walk generators if not already
        if self.list_generators is None:
            self._init_rw_generators()
        
        n = self.config['n_permutations_length']
        self.dqn_model = PermutationMLP(
            input_size=n,
            hidden_dims=self.config['list_layers_sizes'],
            num_classes_for_one_hot=n
        ).to(self.device)
        
        # Copy shared MLP weights into DQN
        mlp_sd = self.mlp_model.state_dict()
        dqn_sd = self.dqn_model.state_dict()
        for name, param in mlp_sd.items():
            if name in dqn_sd:
                # copy_ into the param tensor clone
                dqn_sd[name].copy_(param.to(self.device))
        # Load the updated state dict back into the DQN model
        self.dqn_model.load_state_dict(dqn_sd)

        self.X_anchor, self.y_anchor, self.X_test, self.y_test = self.generate_training_data_anchor()
        
        self.dqn_trainer = DQNTrainer(
            model=self.mlp_trainer.model,
            X_anchor=self.X_anchor,
            y_anchor=self.y_anchor,
            criterion=self.mlp_trainer.criterion,
            optimizer=self.mlp_trainer.optimizer,
            list_generators=self.list_generators,
            tensor_generators=self.tensor_generators,
            cfg=self.config,
            state_destination=self.state_destination,
            random_walks_type=self.config['random_walks_type'],
            device=self.device
        )
    
    def generate_training_data(self):

        n = self.config['n_permutations_length']

        def get_LRX_moves(n):
            L = np.array( list(np.arange(1,n)) + [0])
            R = np.array( [n-1] + list(np.arange(n-1)) )
            X = np.array( [1,0] + list(np.arange(2,n)) )
            return L,R,X
        
        L,R,X = get_LRX_moves(n)
        dict_generators = {'L':L,'R':R,'X':X } 

        list_generators = [L,R,X ]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        dtype_generators = torch.int64  
            
        random_walks_type = self.config['random_walks_type']
        n_random_walk_length = self.config['n_random_walk_length'] 
        n_random_walks_to_generate = self.config['n_random_walks_to_generate'] 
        n_random_walks_steps_back_to_ban = self.config['n_random_walks_steps_back_to_ban']
        state_destination = torch.arange( len(list_generators[0]) , device = device, dtype =  dtype_generators  )

        X,y = random_walks_nbt(state_destination=state_destination, generators=list_generators, 
                n_random_walk_length=n_random_walk_length, n_random_walks_to_generate=n_random_walks_to_generate,
                n_random_walks_steps_back_to_ban=n_random_walks_steps_back_to_ban, random_walks_type=random_walks_type)
        
        return X, y
    
    def generate_training_data_anchor(self, num_of_samples=1_000_000, mode='single'):
        n = self.config['n_permutations_length']
        X,y = bfs_build_dataset(self.state_destination, self.list_generators, self.device, num_of_samples=num_of_samples)
        if mode == 'quadruples':
            X, y = build_quadruples_from_bfs(X, y, self.tensor_generators, self.device)

        # define train and test sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
        return X_train, y_train, X_test, y_test
    
    def train_mlp(self, mode='random_walks'):
        """Train MLP model"""
        if self.mlp_model is None:
            self.setup_mlp_model()
        if mode == 0:
            self.mlp_exp_name = self.get_unique_log_dir()
            print('Training MLP with random walks')
            writer = SummaryWriter(f'{self.mlp_exp_name}')
            writer.add_text("config", json.dumps(self.config, indent=2))

            mlp_losses = []
            n_epochs = self.config['n_epochs']
            pbar = tqdm(range(n_epochs), desc="Training MLP with random walks")
            for epoch in pbar:
                X, y = self.generate_training_data()
                loss = self.mlp_trainer.train_epoch(X, y)
                writer.add_scalar('Loss/train', loss, epoch)
                mlp_losses.append(loss)
                if (epoch + 1) % 10 == 0:
                    # обновляем только postfix, не ломая самую прогресс-строку
                    pbar.set_postfix(loss=f"{loss:.4f}")
                for name, param in self.mlp_model.named_parameters():
                    writer.add_histogram(name, param.data, epoch)

        elif mode == 1:
            self.mlp_exp_name = self.get_unique_log_dir()
            print('Training MLP with anchor')
            writer = SummaryWriter(f'{self.mlp_exp_name}')
            mlp_losses = []
            n_epochs = self.config['n_epochs']
            pbar = tqdm(range(n_epochs), desc="Training MLP with anchor")
            for epoch in pbar:
                loss = self.mlp_trainer.train_epoch(self.X_anchor, self.y_anchor)
                test_loss = self.mlp_trainer.evaluate(self.X_test, self.y_test)
                writer.add_scalar('Loss/train', loss, epoch)
                writer.add_scalar('Loss/test', test_loss, epoch)
                writer.add_scalars('Loss', {
                    'train': loss,
                    'test': test_loss
                }, epoch)
                writer.add_text("config", json.dumps(self.config, indent=2))
                mlp_losses.append(loss)
                if (epoch + 1) % 10 == 0:
                    # обновляем только postfix, не ломая самую прогресс-строку
                    pbar.set_postfix(train_loss=f"{loss:.4f}", test_loss=f"{test_loss:.4f}")

                for name, param in self.mlp_model.named_parameters():
                    writer.add_histogram(name, param.data, epoch)
            writer.close()


        return mlp_losses
    
    def train_dqn(self):
        """
        Train DQN model via the refactored DQNTrainer class.
        """
        if self.dqn_trainer is None:
            self.setup_dqn_model()
        if self.mode == 'single_hard_hinge':
            history = self.dqn_trainer.train_single_hard_hinge()
        elif self.mode == 'double_soft_hinge':
            history = self.dqn_trainer.train_double_soft_hinge()
        return history['train_loss']
    
    def test_beam_search(self):
        state_start, state_destination = initialize_states(self.list_generators, self.device)
        results_df = beam_search_path(self.config, state_start, state_destination, self.list_generators, self.tensor_generators, self.dqn_model, self.device, torch.int64)
        
        return results_df
    
    def main(self):
        self.train_mlp()
        self.train_dqn()
        self.test_beam_search() 