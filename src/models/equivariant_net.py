import torch
import torch.nn as nn

class PE_Layer(nn.Module):
    """
    Permutation-Equivariant Layer по мотивам Maron et al. (2020).

    На входе: тензор x размера (B, n, d_in)
      - B — batch-size
      - n — число “позиционных” элементов (в нашем случае n = 10)
      - d_in — размер embedding для каждой позиции

    На выходе: y размера (B, n, d_out), такой что слой эквивариантен к S_n:
      y_i = A x_i + B (sum_j x_j) + c
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # “локальный” вес: обрабатывает x_i отдельно
        self.A = nn.Linear(d_in, d_out, bias=False)
        # “глобальный” вес: обрабатывает сумму по всем позициям
        self.B = nn.Linear(d_in, d_out, bias=False)
        # свободный вектор смещения
        self.c = nn.Parameter(torch.zeros(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, n, d_in)
        returns: Tensor of shape (B, n, d_out),
                 эквивариантный к перестановкам на оси n.
        """
        # Усреднём глобально по n: shape → (B, 1, d_in)
        # Если нужны суммы, а не среднее, можно использовать x.sum(dim=1, keepdim=True)
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, d_in)

        # A(x_i) — применяется отдельно к каждой «позиции»
        out_local = self.A(x)              # (B, n, d_out)

        # B( sum_j x_j ) — константа вдоль измерения n, потом распространяется
        out_global = self.B(mean)          # (B, 1, d_out)
        out_global = out_global.expand(-1, x.size(1), -1)  # (B, n, d_out)

        # Складываем + смещение c:
        y = out_local + out_global + self.c  # (B, n, d_out)
        return y


class PermutationEquivariantNet(nn.Module):
    def __init__(self, n_positions=10, emb_dim=32, hidden_dim=64, n_pe_layers=3):
        super().__init__()
        self.n = n_positions
        self.emb_dim = emb_dim

        # 1) Сначала каждую позицию “закодируем” через обычный embedding:
        #    вход: целое число 0..9 → выход: вектор размера emb_dim
        self.position_embedding = nn.Embedding(self.n, emb_dim)

        # 2) Несколько эквивариантных слоёв подряд
        self.pe_layers = nn.ModuleList()
        d_in = emb_dim
        for _ in range(n_pe_layers):
            self.pe_layers.append( PE_Layer(d_in=d_in, d_out=hidden_dim) )
            d_in = hidden_dim

        # 3) После этого у нас тензор (B, n, hidden_dim).
        #    Чтобы получить инвариантную фичу, усредним по n:
        #    → (B, hidden_dim)
        # 4) Затем передадим через небольшой MLP, который вернёт скаляр.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x: LongTensor формы (B, n), где x[b,i] ∈ {0..n-1} — значение на позиции i.
        Возвращает: Tensor формы (B, 1) — прогноз расстояния d(π).
        """
        B, n = x.shape
        # 1) Поэлементное embedding: shape (B, n, emb_dim)
        h = self.position_embedding(x)

        # 2) Прокидываем через PE-слои
        for pe in self.pe_layers:
            # h имеет форму (B, n, d_in) → на выходе (B, n, d_out)
            h = pe(h)
            h = torch.relu(h)

        # 3) Аггрегируем: берем среднее (инвариантно к перестановкам)
        h_mean = h.mean(dim=1)  # (B, hidden_dim)

        # 4) Голова: MLP → одно число
        out = self.head(h_mean)  # (B, 1)
        return out
