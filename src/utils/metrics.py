import numpy as np
from typing import List

def hamming_distance(perm1: np.ndarray, perm2: np.ndarray) -> int:
    """
    Calculate Hamming distance between two permutations
    
    Args:
        perm1: First permutation
        perm2: Second permutation
        
    Returns:
        Number of positions where permutations differ
    """
    return np.sum(perm1 != perm2)

def is_sorted(perm: np.ndarray) -> bool:
    """
    Check if permutation is sorted (identity permutation)
    
    Args:
        perm: Permutation to check
        
    Returns:
        True if permutation is sorted
    """
    return np.all(perm == np.arange(len(perm)))

def path_length(path: List[np.ndarray]) -> int:
    """
    Calculate length of path (number of transpositions)
    
    Args:
        path: List of permutations forming a path
        
    Returns:
        Number of transpositions in path
    """
    return len(path) - 1

def validate_path(path: List[np.ndarray]) -> bool:
    """
    Validate that path is valid (each step is a transposition)
    
    Args:
        path: List of permutations forming a path
        
    Returns:
        True if path is valid
    """
    if not path:
        return False
        
    for i in range(len(path) - 1):
        diff = np.sum(path[i] != path[i+1])
        if diff != 2:  # Valid transposition changes exactly 2 positions
            return False
            
    return True 