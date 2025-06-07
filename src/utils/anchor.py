from collections import deque, defaultdict
import torch
import numpy as np
from zmq import device


def bfs_build_dataset(state_destination, list_generators, device, num_of_samples=5_000_000):
    start_state = tuple(state_destination.tolist())
    queue = deque([start_state])
    visited = {start_state: 0}
    
    while queue and len(visited) < num_of_samples:
        current_state = queue.popleft()
        current_depth = visited[current_state]
        
        for gen in list_generators:
            next_state_list = [current_state[gen[i]] for i in range(len(gen))]
            next_state = tuple(next_state_list)
            
            if next_state not in visited:
                visited[next_state] = current_depth + 1
                queue.append(next_state)
    
    all_states = list(visited.keys())
    print(len(all_states))
    depths = [visited[s] for s in all_states]
    
    X = torch.tensor(all_states, dtype=torch.long)
    y = torch.tensor(depths, dtype=torch.long)

    print('X.shape:',X.shape)
    print('y.shape:',y.shape)
    
    return X.to(device), y.to(device)

def build_quadruples_from_bfs(
    X: torch.Tensor,                  # shape = (N, n), dtype=torch.long
    y: torch.Tensor,                  # shape = (N,),   dtype=torch.long
    list_generators: list,            # три тензора, каждый shape = (n,), dtype=long
    device: str = 'cpu'
):
    """
    Из полного BFS-результата (X, y) строит для каждой вершины
    её трёх соседей и сразу выводит глубины этих соседей.

    Возвращает:
      X_quadruples: torch.Tensor shape=(N, 4, n), dtype=torch.long, device=device
      y_quadruples: torch.Tensor shape=(N, 4),    dtype=torch.long, device=device

    где для каждого i in [0..N-1]:
      X_quadruples[i,0] = X[i]
      X_quadruples[i,j] = X[i][ generators[j-1] ]  для j=1,2,3
      y_quadruples[i,0] = y[i]
      y_quadruples[i,j] = глубина вершины X[i] после применения generators[j-1].

    Предполагается, что любые “соседние” состояния действительно лежат 
    в X (то есть вы уже сделали полный BFS и собрали ВСЕ достижимые узлы).
    """

    # 1) Параметры
    X = X.to(device)   # (N, n)
    y = y.to(device)   # (N,)
    N, n = X.shape
    gens = [g.to(device).long() for g in list_generators]
    assert len(gens) == 3, "Ожидается ровно 3 генератора"

    # 2) Вычислим “ID” для каждой строки X[i].
    #    Поскольку n = 10 и значения в каждой строке – перестановка из [0..9], 
    #    мы можем “упаковать” её в int64 как sum_j X[i,j]*10^j.
    #    (Обратите внимание: 10^10 < 2^63, так что в int64 всё влезает.)
    #
    #    Сначала сформируем вектор “powers_of_ten” длины n:

    powers = (n ** torch.arange(n, dtype=torch.int64, device=device))  # shape=(n,)
    #    Затем вычисляем ids_main[i] = sum_j X[i,j] * 10^j
    ids_main = (X * powers.unsqueeze(0)).sum(dim=1)  # shape=(N,), dtype=int64

    # 3) Сортируем ids_main, чтобы получить sorted_ids и соответствующие глубины sorted_y
    sorted_ids, sort_idx = ids_main.sort()              # оба shape=(N,)
    sorted_y = y[sort_idx]                              # shape=(N,)

    # 4) Теперь для каждого из трёх генераторов вычислим “соседнюю” матрицу:
    #    X_neig_j = X[:, gens[j]]  shape = (N, n)
    X_neig_1 = X[:, gens[0]]
    X_neig_2 = X[:, gens[1]]
    X_neig_3 = X[:, gens[2]]

    # 5) Вычислим для каждой “матрицы соседей” её ID-ы точно так же:
    ids_neig_1 = (X_neig_1 * powers.unsqueeze(0)).sum(dim=1)  # (N,)
    ids_neig_2 = (X_neig_2 * powers.unsqueeze(0)).sum(dim=1)
    ids_neig_3 = (X_neig_3 * powers.unsqueeze(0)).sum(dim=1)

    # 6) Теперь, чтобы узнать глубину конкретного “ID” соседа, 
    #    мы воспользуемся тем, что sorted_ids — отсортирован. 
    #    Для каждого элемента id_neig мы найдём позицию pos = searchsorted(sorted_ids, id_neig).
    #    Поскольку id_neig гарантированно есть в sorted_ids, 
    #    у нас должно получиться sorted_ids[pos] == id_neig, и тогда 
    #    глубина соседа = sorted_y[pos].
    idx1 = torch.searchsorted(sorted_ids, ids_neig_1)  # shape=(N,)
    idx2 = torch.searchsorted(sorted_ids, ids_neig_2)
    idx3 = torch.searchsorted(sorted_ids, ids_neig_3)

    # 7) Берём сами глубины “соседей” из sorted_y:
    y_neig_1 = sorted_y[idx1]  # shape=(N,)
    y_neig_2 = sorted_y[idx2]
    y_neig_3 = sorted_y[idx3]

    # 8) Теперь соберём итоговые “четвёрки” (X_quadruples, y_quadruples).
    #    X_quadruples[i] должен быть [X[i], X_neig_1[i], X_neig_2[i], X_neig_3[i]].
    #    Создадим тензор-заготовку размером (N, 4, n):
    X_quadruples = torch.empty((N, 4, n), dtype=torch.long, device=device)
    y_quadruples = torch.empty((N, 4), dtype=torch.long, device=device)

    # 8.1) Нулевой “столбец” — это просто X и y:
    X_quadruples[:, 0, :] = X             # (N, n)
    y_quadruples[:, 0]    = y             # (N,)

    # 8.2) Первый сосед:
    X_quadruples[:, 1, :] = X_neig_1
    y_quadruples[:, 1]    = y_neig_1

    # 8.3) Второй сосед:
    X_quadruples[:, 2, :] = X_neig_2
    y_quadruples[:, 2]    = y_neig_2

    # 8.4) Третий сосед:
    X_quadruples[:, 3, :] = X_neig_3
    y_quadruples[:, 3]    = y_neig_3

    B, four, state_size = X_quadruples.shape
    

    return X_quadruples, y_quadruples



def get_LRX_moves(n):
    L = np.array( list(np.arange(1,n)) + [0])
    R = np.array( [n-1] + list(np.arange(n-1)) )
    X = np.array( [1,0] + list(np.arange(2,n)) )
    return L,R,X