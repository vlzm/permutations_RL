import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F

class PermutationMLP(nn.Module):
    """
        layer_sizes (list of int): 
            A list defining the number of neurons in each hidden layer. The length of the list determines 
            the number of hidden layers, and each element specifies the size of that layer.
            Example: [32, 16, 8] creates 3 hidden layers with 32, 16, and 8 neurons respectively and nn.ReLU() activations between.
        """
    def __init__(self, input_size, hidden_dims, num_classes_for_one_hot):
        super(PermutationMLP, self).__init__()
        self.num_classes_for_one_hot = num_classes_for_one_hot
        self.input_layer_size_for_one_hot = input_size * num_classes_for_one_hot
        
        layers = []
        in_features = self.input_layer_size_for_one_hot
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):            
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.num_classes_for_one_hot) \
        .float().flatten(start_dim=-2)
        return self.layers(x)
    

class PermutationQuadMLP(nn.Module):
    """
    Модифицированная MLP, которая принимает на вход «четвёрки» состояний (B, 4, n) 
    вместо одиночных (B, n). Внутри один и тот же MLP-энкодер применяется к каждому из 4 состояний,
    а на выходе получается тензор формы (B, 4) с предсказанными значениями для каждого из них.

    Параметры:
        input_size (int): длина каждой перестановки (n).
        hidden_dims (list of int): список размеров скрытых слоёв, например [32, 16, 8].
        num_classes_for_one_hot (int): число классов для one-hot кодирования каждого элемента перестановки.
    """
    def __init__(self, input_size: int, hidden_dims: list[int], num_classes_for_one_hot: int):
        super(PermutationQuadMLP, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes_for_one_hot

        # Размер входного слоя после one-hot: n * num_classes
        self.input_one_hot_dim = input_size * num_classes_for_one_hot

        # Собираем список слоёв MLP (shared для каждого из 4 состояний)
        layers = []
        in_features = self.input_one_hot_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h

        # Финальный линейный слой возвращает скаляр для каждого состояния
        layers.append(nn.Linear(in_features, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: LongTensor формы (B, 4, n), где B — batch size, n — размер перестановки.
        Возвращает FloatTensor формы (B, 4) — предсказанные значения для каждой из 4 позиций.
        """
        B, k, n = x.shape
        assert k == 4 and n == self.input_size, \
            f"Ожидалось x.shape = (B, 4, {self.input_size}), получили {x.shape}"

        # 1) One-hot кодируем все 4 состояния сразу: (B, 4, n) → (B, 4, n, num_classes)
        x_onehot = F.one_hot(x.long(), num_classes=self.num_classes).float()

        # 2) Плоское объединение последних двух размерностей: (B, 4, n, C) → (B, 4, n*C)
        x_flat4 = x_onehot.view(B, k, n * self.num_classes)

        # 3) Сразу превращаем (B, 4, n*C) → (B*4, n*C), чтобы пропустить через единый MLP
        x_flat = x_flat4.view(B * k, n * self.num_classes)

        # 4) Forward-проход через MLP: (B*4, n*C) → (B*4, 1)
        out_flat = self.mlp(x_flat)              # shape = (B*4, 1)

        # 5) Сжимаем размер (B*4, 1) → (B*4,) и затем раскладываем в (B, 4)
        out = out_flat.view(B, k)

        return out

class EquivariantBlock(nn.Module):
    """
    Один permutation-equivariant блок для одной перестановки размера n:
      x_onehot: (B, n, C) → 
      φ: (C → d) applied to each of n positions → (B, n, d)
      u = sum_i φ(x_i) → (B, d)
      ψ: (2d → d) applied to each of n positions → (B, n, d)
    """
    def __init__(self, num_classes, d):
        super().__init__()
        self.phi = nn.Linear(num_classes, d, bias=True)
        self.psi = nn.Sequential(
            nn.Linear(2*d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        # x_onehot: (B, n, C)
        # φ на каждую позицию
        phi_out = self.phi(x_onehot)             # (B, n, d)
        # агрегат
        u = phi_out.sum(dim=1, keepdim=True)    # (B, 1, d)
        # конкатенация [φ, u]
        u_rep = u.expand(-1, phi_out.size(1), -1)  # (B, n, d)
        concat = torch.cat([phi_out, u_rep], dim=-1)  # (B, n, 2d)
        # ψ обратно на (B, n, d)
        h = self.psi(concat)                    # (B, n, d)
        return h

class PermutationQuadEquivariantMLP(nn.Module):
    """
    Обобщение на ваши «четвёрки»: для каждой из 4 перестановок сначала DeepSets-блок,
    потом агрегируем n→скаляр, и собираем обратно (B,4).
    """
    def __init__(self, n: int, num_classes: int, hidden_dims: list[int]):
        """
        n: длина перестановки
        num_classes: C
        hidden_dims: список d-ок, например [64, 32] — глубина каждого φ/ψ-блока
        """
        super().__init__()
        self.n = n
        self.C = num_classes
        # первый φ/ψ-блок
        d0 = hidden_dims[0]
        self.eq_block = EquivariantBlock(self.C, d0)

        # дополнительные DeepSets-блоки (по желанию)
        self.extra_blocks = nn.ModuleList([
            EquivariantBlock(d0, d1) for d1 in hidden_dims[1:]
        ])

        # из d_last → скаляр
        self.to_scalar = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: LongTensor (B, 4, n)
        return: FloatTensor (B, 4)
        """
        start_time = time.time()
        B, K, n = x.shape
        assert K == 4 and n == self.n
        # print(f"Time for shape assertion: {time.time() - start_time} seconds")

        start_time = time.time()
        x_oh = F.one_hot(x, num_classes=self.C).float()
        x_oh = x_oh.view(B*K, n, self.C)
        # print(f"Time for one-hot encoding and reshaping: {time.time() - start_time} seconds")

        start_time = time.time()
        h = self.eq_block(x_oh)
        # print(f"Time for eq_block: {time.time() - start_time} seconds")

        for blk in self.extra_blocks:
            # print("Start blk")
            start_time = time.time()
            h = blk(h)
            # print(f"Time for extra block: {time.time() - start_time} seconds")

        start_time = time.time()
        # агрегируем по n к вектору (B*4, d_last)
        h_sum = h.sum(dim=1)                   # (B*4, d_last)
        # print(f"Time for summing h: {time.time() - start_time} seconds")

        start_time = time.time()
        # в скаляр
        out = self.to_scalar(h_sum).view(B, K) # (B,4)
        # print(f"Time for to_scalar: {time.time() - start_time} seconds")

        return out
    
class DeepSetMLP(nn.Module):
    """
    DeepSets-подход для одной перестановки (B, n) → скаляр (B, 1),
    где weight-sharing по всем n позициям гарантирует, что модель
    не «запоминает» (позиция i, метка c) как отдельный признак.
    """
    def __init__(self, n: int, num_classes: int, phi_dim: int, mlp_hidden: list[int]):
        """
        Args:
          n            — размер перестановки
          num_classes  — число меток (обычно == n)
          phi_dim      — размер φ-эмбеддинга каждой позиции
          mlp_hidden   — список скрытых слоёв после агрегата
        """
        super().__init__()
        # φ: one-hot(c) → R^{phi_dim}, shared для всех позиций
        self.phi = nn.Linear(num_classes, phi_dim)
        # после суммы по позициям, MLP: phi_dim → ... → 1
        layers = []
        in_f = phi_dim
        for h in mlp_hidden:
            layers += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        layers.append(nn.Linear(in_f, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: LongTensor (B, n)
        returns: FloatTensor (B, 1)
        """
        B, n = x.shape
        # 1) one-hot: (B,n) → (B,n,num_classes)
        x_oh = F.one_hot(x, num_classes=self.phi.in_features).float()
        # 2) φ на каждую позицию: (B,n,C) → (B,n,phi_dim)
        h = self.phi(x_oh)                  # shared weights
        # 3) агрегируем по позициям: (B,n,phi_dim) → (B,phi_dim)
        u = h.sum(dim=1)
        # 4) head → (B,1)
        return self.head(u)
    
class EquivariantSumMLP(nn.Module):
    """
    Shared embedding по меткам  +  learnable positional weights.
    Выход зависит от того, какая метка оказалась в какой позиции,
    но веса φ всё-таки общие для позиций  -> weight sharing сохранён.
    """
    def __init__(self, n: int, num_classes: int, phi_dim: int, mlp_hidden: list[int]):
        super().__init__()
        self.n = n
        self.C = num_classes

        # φ: one-hot(value) -> R^d  (shared для всех позиций)
        self.val_emb = nn.Linear(num_classes, phi_dim, bias=False)

        # позиционные веса  (по одному вектору на позицию i)
        self.pos_emb = nn.Parameter(torch.randn(n, phi_dim))

        # head: (phi_dim) -> ... -> 1
        layers = []
        in_f = phi_dim
        for h in mlp_hidden:
            layers += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        layers.append(nn.Linear(in_f, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):                          # x: (B, n)
        B, n = x.shape
        assert n == self.n

        # one-hot -> shared embedding: (B,n,C) -> (B,n,d)
        h_val = self.val_emb(F.one_hot(x, self.C).float())

        # добавляем позиционный вес: broadcast (n,d) -> (B,n,d)
        h = h_val * self.pos_emb                 # element-wise

        # суммируем по позициям -> (B,d)
        u = h.sum(dim=1)

        # карта -> скаляр
        return self.head(u)                      # shape (B,1)