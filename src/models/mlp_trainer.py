import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from tqdm import tqdm
import torch.nn.functional as F

class QuadMLPTrainer:
    def __init__(self, model, config, tensor_generators):
        """
        model: ваша PermutationQuadMLP (принимает вход (B, 4, n) и выдаёт (B, 4))
        config: словарь с параметрами, ожидаем, что в нём есть:
            - 'batch_size': число «четвёрок» в одном батче
            - 'learning_rate'
            - опционно 'lambda_smooth' (вес для loss на разности соседей)
        tensor_generators: список тензоров-перестановок (не нужен напрямую для train_epoch, но храним для интерфейса)
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Размер батча в терминах «четвёрок»
        self.batch_size_quads = config.get('batch_size', 256)
        self.learning_rate = config.get('learning_rate', 1e-4)
        # Веса для регуляризации гладкости (разность между первым элементом и остальными)
        self.lambda_smooth = config.get('lambda_smooth', 1)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss().to(self.device)
        
        self.tensor_generators = tensor_generators
        self.model.to(self.device)

    def train_epoch(self, X_quadruples, y_quadruples):
        """
        X_quadruples: torch.Tensor, shape = (N, 4, n), dtype=torch.long
        y_quadruples: torch.Tensor, shape = (N, 4), dtype=torch.long или float
        Возвращает: средний комбинированный loss по всем батчам за эпоху.
        """
        # perm_raw = torch.randperm(X_quadruples.shape[0], device='cpu')[:self.batch_size_quads*100]
        # X_quadruples = X_quadruples[perm_raw]
        # y_quadruples = y_quadruples[perm_raw]

        N, four, n = X_quadruples.shape
        assert four == 4, "Ожидаем, что второй размер X_quadruples == 4"
        assert y_quadruples.shape == (N, 4)
        
        # Переводим на нужное устройство
        X = X_quadruples.to(self.device)
        y = y_quadruples.to(self.device).float()
        
        # Перевешиваем индексы «четвёрок»
        perm = torch.randperm(N, device='cpu')
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for start in range(0, N, self.batch_size_quads):
            end = min(start + self.batch_size_quads, N)
            batch_idx = perm[start:end]
            # print("Start batch")
            start_time = time.time()
            Xb = X[batch_idx]   # shape = (B, 4, n)
            yb = y[batch_idx]   # shape = (B, 4)
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")
            
            # print("Start model")
            start_time = time.time()
            # Прямой проход: модель выдаёт (B, 4)
            preds = self.model(Xb)  # shape = (B, 4)
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")
            
            # print("Start loss")
            start_time = time.time()
            # Основной MSE loss между предсказаниями и y
            loss_main = self.criterion(preds, yb)
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")
            
            # Smoothness loss: хотим, чтобы pred[:,0] - pred[:,i] ≈ y[:,0] - y[:,i] для i=1..3
            # Вычисляем разности
            # print("Start smoothness loss")
            start_time = time.time()
            y_center = yb[:, 0]
            y_neighbors = yb[:, 1:]
            delta_true = y_center.unsqueeze(1) - y_neighbors
            pred_center = preds[:, 0]
            pred_neighbors = preds[:, 1:]
            delta_pred = pred_center.unsqueeze(1) - pred_neighbors
            loss_smooth = self.criterion(delta_pred, delta_true)
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")

            # loss_temp = self.criterion(preds[:, 0], yb[:, 0])
            # print("Start total loss")
            start_time = time.time()
            loss = loss_main + self.lambda_smooth * loss_smooth
            # print(f"Loss: {loss.item()}")
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")
            # loss = loss_temp
            
            
            # Обратный проход
            self.optimizer.zero_grad()
            # print("Start backward")
            start_time = time.time()
            loss.backward()
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")
            # print("Start step")
            start_time = time.time()
            self.optimizer.step()
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")
            # print("End step")
            total_loss += loss.item()
            num_batches += 1
            # print("End batch")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
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
        # outputs_start = self.model(X)
        # print(outputs_start.squeeze())
        # print(y.float())
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
