from collections import deque, defaultdict
import torch
import numpy as np

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


def get_LRX_moves(n):
    L = np.array( list(np.arange(1,n)) + [0])
    R = np.array( [n-1] + list(np.arange(n-1)) )
    X = np.array( [1,0] + list(np.arange(2,n)) )
    return L,R,X