import torch
import numpy as np
import time
from typing import List, Union, Optional, Tuple, Any
from dataclasses import dataclass
from src.utils.random_walks import get_neighbors
import pandas as pd

def initialize_states(list_generators, device):
    n = len(list_generators[0])
    p = np.arange(n)
    # Swap (0,2)
    p[0], p[1] = p[1], p[0]
    i = 2
    while i < n-i+1:
        #print(i, n-i+1)
        p[i], p[n-i+1] = p[n-i+1], p[i]
        i += 1
    permutation_longest = torch.tensor(p, dtype=torch.int64, device=device)
    state_start = permutation_longest

    state_destination = torch.arange(len(list_generators[0]), device=device, dtype=torch.int64)
    
    return state_start, state_destination

def get_unique_states(states: torch.Tensor, vec_hasher = 'Auto',device = 'Auto') -> torch.Tensor:
    '''
    Return matrix with unique rows for input matrix "states" 
    I.e. duplicate rows are dropped.
    For fast implementation: we use hashing via scalar/dot product.
    Note: output order of rows is different from the original. 

    Note: that implementation is 30 times faster than torch.unique(states, dim = 0) - because we use hashes  (see K.Khoruzhii: https://t.me/sberlogasci/10989/15920)
    Note: torch.unique does not support returning of indices of unique element so we cannot use it 
    That is in contrast to numpy.unique which supports - set: return_index = True     
    '''

    if vec_hasher == 'Auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
    # Preprare vector which will used for hashing - to avoid revisiting same states 
    if vec_hasher == 'Auto':
        max_int =  int( (2**62) )
        dtype_for_hash = torch.int64
        vec_hasher = torch.randint(-max_int, max_int+1, size=(len(states[0]),), device=device, dtype=dtype_for_hash)     
    
    # Hashing rows of states matrix: 
    hashed = torch.sum(vec_hasher * states, dim=1) # Compute hashes. 
        # It is same as matrix product torch.matmul(hash_vec , states ) 
        # but pay attention: such code work with GPU for integers 
        # While torch.matmul - does not work for GPU for integer data types, 
        # since old GPU hardware (before 2020: P100, T4) does not support integer matrix multiplication 
    
    # Sort
    hashed_sorted, idx = torch.sort(hashed)
    # Mask selects elements which are different from the consequite - that is unique elements (since vector is sorted on the previous step)
    mask = torch.concat((torch.tensor([True], device = device), 
                         (hashed_sorted[1:] - hashed_sorted[:-1]) != 0))
    return states[idx][mask]

def beam_search_torch(CFG, state_start, state_destination, list_generators, tensor_generators, model, device, dtype):
    if not CFG['beam_search_torch']:
        return

    # To implement ban_p0_p1_transposition_if_p0_lt_p1_ie_already_sorted
    # We need the following to known where what is the position of transposition (0,1) in list_generators
    #X_loc = torch.tensor( [1,0] + list(np.arange(2,state_size)), device = device, dtype = dtype )

    ##########
    X_loc = np.array( [1,0] + list(np.arange(2,len(list_generators[0]))) )
    i_position_X_in_list_generators = -1
    for k in range(len(list_generators)):
        if np.all( list_generators[k] == X_loc ) : 
            i_position_X_in_list_generators = k
            break
    print('i_position_X_in_list_generators:',i_position_X_in_list_generators )  

    ##########
    state_size = len(list_generators[0] )
    max_int =  int( (2**62) )
    dtype_for_hash = torch.int64
    # if vec_hasher == 'Auto':
    vec_hasher = torch.randint(-max_int, max_int+1, size=(state_size,), device=device, dtype=dtype_for_hash).to(device)
    ##########

    n_generators = len(list_generators)
    t0 = time.time()
    flag_found_destination = False

    # Initialize array of states 
    array_beam_states = state_start.view(1, len(list_generators[0])).clone().to(dtype).to(device)

    # Hash initial states. Initialize storage
    if CFG['n_beam_search_steps_back_to_ban'] > 0:
        hash_initial_state = torch.sum(state_start.view(-1, len(list_generators[0])) * vec_hasher, dim=1)
        vec_hashes_current = hash_initial_state.expand(CFG['beam_width'] * n_generators, CFG['n_beam_search_steps_back_to_ban']).clone()
        i_cyclic_index_for_hash_storage = 0

    for i_step in range(1, CFG['n_steps_limit'] + 1):
        t_moves = t_hash = t_isin = 0
        t_full_step = time.time()
        t_unique_els = 0

        # Create new states
        t1 = time.time()
        if not CFG['ban_p0_p1_transposition_if_p0_lt_p1_ie_already_sorted']:
            array_new_states = get_neighbors(array_beam_states, tensor_generators).flatten(end_dim=1)
        else:
            array_new_states = torch.empty((0, array_beam_states.shape[1]), device=device, dtype=dtype)
            row_indices = np.arange(array_beam_states.shape[0])[:, np.newaxis]
            for ii1, move in enumerate(list_generators):
                if ii1 != i_position_X_in_list_generators:
                    array_states_tmp = array_beam_states[row_indices, move]
                else:
                    mask_X_condtion     = array_beam_states[:, 0] > array_beam_states[:, 1]
                    row_indices = np.arange(mask_X_condtion.sum().item())[:, np.newaxis]
                    array_states_tmp = array_beam_states[mask_X_condtion][row_indices, move]
                array_new_states = torch.concatenate([array_new_states, array_states_tmp], axis=0)
        t_moves += (time.time() - t1)

        # Take only unique states
        t1 = time.time()
        array_new_states = get_unique_states(array_new_states)
        t_unique_els += (time.time() - t1)

        # Check destination state found
        vec_tmp = torch.all(array_new_states == state_destination, axis=1)
        flag_found_destination = torch.any(vec_tmp).item()
        if flag_found_destination:
            if CFG['verbose'] >= 1:
                print('Found destination state. ', 'i_step:', i_step, ' n_ways:', (vec_tmp).sum())
            break

        # Nonbacktracking - forbid visits states visited before
        if CFG['n_beam_search_steps_back_to_ban'] > 0:
            t1 = time.time()
            vec_hashes_new = torch.sum(array_new_states * vec_hasher, dim=1)
            t_hash += (time.time() - t1)

            t1 = time.time()
            mask_new = ~torch.isin(vec_hashes_new, vec_hashes_current.view(-1), assume_unique=False)
            t_isin += (time.time() - t1)
            mask_new_sum = mask_new.sum().item()
            if mask_new_sum > 0:
                array_new_states = array_new_states[mask_new, :]
            else:
                flag_found_destination = False
                if CFG['verbose'] >= 1:
                    print('Cannot find new states. i_step:', i_step)
                break
            i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1) % CFG['n_beam_search_steps_back_to_ban']
            i_tmp = len(vec_hashes_new)
            vec_hashes_current[:i_tmp, i_cyclic_index_for_hash_storage] = vec_hashes_new

        # Estimate states and select top beam_width ones
        t0 = time.time()
        q_value = torch.tensor([0])
        if array_new_states.shape[0] > CFG['beam_width']:
            if CFG['beam_search_models_or_heuristics'] == 'model_torch':
                model.eval()
                with torch.no_grad():
                    n_states_all = array_new_states.shape[0]
                    q_value = torch.zeros(n_states_all, device=device)
                    for i_start_batch in range(0, n_states_all, CFG['batch_size']):
                        i_end_batch = min([i_start_batch + CFG['batch_size'], n_states_all])
                        q_value[i_start_batch:i_end_batch] = model(array_new_states[i_start_batch:i_end_batch, :]).view(-1)
                idx = torch.argsort(q_value)[:CFG['beam_width']]
                array_beam_states = array_new_states[idx, :]
            elif CFG['beam_search_models_or_heuristics'] == 'model_with_predict':
                q_value = model.predict(array_new_states.detach().cpu().numpy())
                idx = np.argsort(q_value)[:CFG['beam_width']]
                array_beam_states = array_new_states[idx, :]
            elif CFG['beam_search_models_or_heuristics'] == 'Hamming':
                q_value = torch.sum((array_new_states - state_destination) != 0, axis=0)
                idx = torch.argsort(q_value)[:CFG['beam_width']]
                array_beam_states = array_new_states[idx, :]
            else:
                raise Exception("Unrecognized models_or_heuristics: " + str(CFG['beam_search_models_or_heuristics']))
        else:
            array_beam_states = array_new_states
        predict_time = time.time() - t0

        if CFG['verbose'] >= 10:
            if (i_step - 1) % 10 == 0:
                if isinstance(q_value, torch.Tensor):
                    q_value = q_value.detach().cpu().numpy()
                t_full_step = time.time() - t_full_step
                print('Step:', i_step, 'Beam (not cumulative) min:', '%.2f' % np.min(q_value),
                      'median:', '%.2f' % np.median(q_value), 'max:', '%.2f' % np.max(q_value))
        if CFG['verbose'] >= 100:
            if (i_step - 1) % 15 == 0:
                print('Time: %.1f' % (time.time() - t0), 't_moves  %.3f, t_hash  %.3f, t_isin %.3f, t_unique_els  %.3f, t_full_step %.3f' % (
                    t_moves, t_hash, t_isin, t_unique_els, t_full_step))

    print()
    print(CFG)
    print()
    print('beam_width:', CFG['beam_width'])
    print('n=', len(list_generators[0]))
    print('n(n-1)/2=', int(len(list_generators[0]) * (len(list_generators[0]) - 1) / 2))
    print('Found Path Length:', i_step, 'flag_found_destination:', flag_found_destination)

    return i_step, flag_found_destination