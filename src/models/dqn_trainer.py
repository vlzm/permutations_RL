import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
import copy
from tqdm import tqdm
from src.utils.random_walks import random_walks_nbt, get_neighbors

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
                - batch_size
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

    def train_double_soft_hinge(self):
        """
        Main training loop for DQN with hinge-Bellman loss and replay buffer.
        """
        n_epochs = self.cfg['n_epochs_dqn']
        verbose = self.cfg.get('verbose_loc', 10)
        buffer_size = self.cfg.get('replay_buffer_size', 200_000)
        batch_size = self.cfg['batch_size']
        replay_start_size = self.cfg.get('replay_start_size', 5000)

        # Create replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        history = {'train_loss': []}
        start_time = time.time()
        print(f"Starting DQN training for {n_epochs} epochs...")

        for epoch in tqdm(range(n_epochs), desc="Training DQN"):
            t0 = time.time()
            # --- 1. Сгенерировать новый набор блужданий ---
            X_new, _ = self._generate_data()  # y_train не нужен!
            
            # --- 2. Добавить их в replay buffer ---
            with torch.no_grad():
                neighbors = self._compute_neighbors(X_new)
                h_neigh = self.target_model(neighbors).view(X_new.size(0), -1)
                upper_bounds = 1 + torch.min(h_neigh, dim=1)[0]  # Upper bounds: (B,)
            
            # for x, upper in zip(X_new, upper_bounds):
            #     self.replay_buffer.append((x.cpu(), upper.cpu()))
            
            X_cpu = X_new.detach().cpu()
            upper_cpu = upper_bounds.detach().cpu()
            self.replay_buffer.extend(zip(X_cpu, upper_cpu))

            t_rw = time.time() - t0

            # --- 3. Обучение ---
            t0 = time.time()
            self.model.train()
            total_loss = 0.0
            count = 0

            # Количество батчей в эпохе — примерно весь новый набор пройти
            n_batches = max(1, len(X_new) // batch_size)

            for _ in range(n_batches):
                if len(self.replay_buffer) < replay_start_size:
                    # Пока буфер мал, учим только на свежем X_new
                    idx = torch.randint(0, X_new.size(0), (batch_size,))
                    batch_X = X_new[idx].to(self.device)
                else:
                    samples = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
                    batch = [self.replay_buffer[i] for i in samples]
                    batch_X, batch_upper = zip(*batch)
                    batch_X = torch.stack(batch_X).to(self.device)
                    batch_upper = torch.stack(batch_upper).to(self.device)

                # --- 4. Прямой проход ---
                h_s = self.model(batch_X).squeeze()

                with torch.no_grad():
                    neighbors = self._compute_neighbors(batch_X)
                    h_neigh = self.target_model(neighbors).view(batch_X.size(0), -1)
                    upper = 1 + torch.min(h_neigh, dim=1)[0]

                # --- 5. Hinge loss ---
                hinge = torch.relu(h_s - upper).mean()
                tight = torch.relu(upper - h_s).mean()

                # --- 6. Anchor MSE (если нужно) ---
                anchor = 0.0
                if self.w_anchor > 0:
                    samples = np.random.choice(len(self.X_anchor), batch_size, replace=False)
                    h_anchor = self.model(self.X_anchor[samples]).squeeze()
                    batch_upper_anchor = self.y_anchor[samples]
                    # Если batch_upper есть (например, при replay_buffer обучении)
                    if len(batch_upper) > 0:
                        anchor = torch.nn.functional.mse_loss(h_anchor, batch_upper_anchor.float(), reduction='mean')

                hinge_loss = self.w_hinge * hinge
                anchor_loss = self.w_anchor * anchor
                tight_loss = self.w_tight * tight

                loss = hinge_loss + anchor_loss + tight_loss

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

            train_loss = total_loss / count
            history['train_loss'].append(train_loss)
            t_train = time.time() - t0

            # --- 7. Обновление target сети ---
            if (epoch + 1) % self.sync_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            # tau = 0.05   # или 0.01–0.1
            # for p_t, p in zip(self.target_model.parameters(),
            #                 self.model.parameters()):
            #     p_t.data.mul_(1 - tau).add_(tau * p.data)

            # --- 8. Логирование ---
            if epoch % verbose == 0:
                print(
                    f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                    f"Hinge: {hinge_loss:.4f} | Anchor: {anchor_loss:.4f} | Tight: {tight_loss:.4f} | "
                    f"Times - RW: {t_rw:.2f}s, Train: {t_train:.2f}s | "
                    f"Buffer size: {len(self.replay_buffer)}"
                )

        print(f"Training finished in {time.time() - start_time:.1f}s")
        return history
    
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
            for start in range(0, n_states, self.cfg['batch_size']):
                end = min(start + self.cfg['batch_size'], n_states)
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
        n_epochs = self.cfg['n_epochs_dqn']
        verbose = self.cfg.get('verbose_loc', 10)

        history = {'train_loss': []}
        start_time = time.time()
        print(f"Starting DQN training for {n_epochs} epochs...")

        for epoch in tqdm(range(n_epochs), desc="Training DQN"):

            if self.w_anchor > 0:
                with torch.no_grad():
                    # закешировать ВСЕ фичи якоря
                    self.h_anchor_all = self.model(self.X_anchor).squeeze()

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
            for start in range(0, n_states, self.cfg['batch_size']):
                end = min(start + self.cfg['batch_size'], n_states)
                batch_X = X_train[start:end]
                batch_y = y_train[start:end]

                outputs = self.model(batch_X)
                loss_hinge = self.criterion(outputs.squeeze(), batch_y)

                anchor = 0.0
                if self.w_anchor > 0:
                    idx = torch.randint(0, len(self.X_anchor), (batch_X.size(0),), device=self.device)
                    h_anchor = self.h_anchor_all[idx]
                    batch_upper_anchor = self.y_anchor[idx]
                    anchor = torch.nn.functional.mse_loss(h_anchor, batch_upper_anchor.float(), reduction='mean')

                loss = self.w_hinge * loss_hinge + self.w_anchor * anchor

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

            train_loss = total_loss / count

            history['train_loss'].append(train_loss)
            t_train = time.time() - t0

            if epoch % verbose == 0:
                print(
                    f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                    f"Hinge: {loss_hinge:.4f} | Anchor: {anchor:.4f} | "
                    f"Times - RW: {t_rw:.2f}s, Bellman: {t_bellman:.2f}s, Train: {t_train:.2f}s"
                )

        print(f"Training finished in {time.time() - start_time:.1f}s")
        return history
    
    #########################################################
    #########################################################
    #########################################################

    def train_single_soft_hinge(self):
        """
        Main training loop for DQN.
        """
        n_epochs = self.cfg['n_epochs_dqn']
        verbose = self.cfg.get('verbose_loc', 10)

        history = {'train_loss': []}
        start_time = time.time()
        print(f"Starting DQN training for {n_epochs} epochs...")

        for epoch in tqdm(range(n_epochs), desc="Training DQN"):
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
            for start in range(0, n_states, self.cfg['batch_size']):
                end = min(start + self.cfg['batch_size'], n_states)
                batch_X = X_train[start:end]
                batch_y = y_train[start:end]

                outputs = self.model(batch_X)
                loss_hinge = self.criterion(outputs.squeeze(), batch_y)

                anchor = 0.0
                if self.w_anchor > 0:
                    samples = np.random.choice(len(self.X_anchor), self.cfg['batch_size'], replace=False)
                    h_anchor = self.model(self.X_anchor[samples]).squeeze()
                    batch_upper_anchor = self.y_anchor[samples]
                    anchor = torch.nn.functional.mse_loss(h_anchor, batch_upper_anchor.float(), reduction='mean')

                loss = self.w_hinge * loss_hinge + self.w_anchor * anchor

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

            train_loss = total_loss / count

            history['train_loss'].append(train_loss)
            t_train = time.time() - t0

            if epoch % verbose == 0:
                print(
                    f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                    f"Hinge: {loss_hinge:.4f} | Anchor: {anchor:.4f} | "
                    f"Times - RW: {t_rw:.2f}s, Bellman: {t_bellman:.2f}s, Train: {t_train:.2f}s"
                )

        print(f"Training finished in {time.time() - start_time:.1f}s")
        return history