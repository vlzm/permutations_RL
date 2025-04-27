import numpy as np
from typing import List, Tuple

class RandomWalkGenerator:
    def __init__(self, n_permutations_length: int, steps_back_to_ban: int = 8):
        """
        Generator for random walks on permutation groups
        
        Args:
            n_permutations_length (int): Length of permutations
            steps_back_to_ban (int): Number of steps back to ban in non-backtracking walks
        """
        self.n = n_permutations_length
        self.steps_back_to_ban = steps_back_to_ban
        
    def generate_simple_walk(self, length: int) -> List[np.ndarray]:
        """Generate simple random walk"""
        perm = np.arange(self.n)
        walk = [perm.copy()]
        
        for _ in range(length):
            i, j = np.random.choice(self.n, size=2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
            walk.append(perm.copy())
            
        return walk
    
    def generate_non_backtracking_walk(self, length: int) -> List[np.ndarray]:
        """Generate non-backtracking random walk"""
        perm = np.arange(self.n)
        walk = [perm.copy()]
        recent_moves = []
        
        for _ in range(length):
            while True:
                i, j = np.random.choice(self.n, size=2, replace=False)
                move = (i, j)
                
                # Check if move would reverse a recent move
                if not any(self._is_reverse_move(move, prev_move) 
                         for prev_move in recent_moves[-self.steps_back_to_ban:]):
                    break
                    
            perm[i], perm[j] = perm[j], perm[i]
            walk.append(perm.copy())
            recent_moves.append(move)
            
        return walk
    
    def _is_reverse_move(self, move1: Tuple[int, int], move2: Tuple[int, int]) -> bool:
        """Check if two moves are reverse of each other"""
        return move1[0] == move2[1] and move1[1] == move2[0] 