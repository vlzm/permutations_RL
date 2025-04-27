import torch
import numpy as np
import time

def get_neighbors(states, moves):
    """
    Some torch magic to apply all moves to all states at once 
    Input:
    states: 2d torch array n_states x n_state_size - rows are states-vectors
    moves (int64): 2d torch array n_moves x  n_state_size - rows are permutations describing moves
    Returns:
    3d tensor all moves applied to all states, shape: n_states x n_moves x n_state_size
    Typically output is followed by .flatten(end_dim=1), which flattens to 2d array ( n_states * n_moves) x n_state_size    
    """
    return torch.gather(
        states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)), 
        2, 
        moves.unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1)))

def random_walks(state_destination, generators , n_random_walk_length,  n_random_walks_to_generate,    state_rw_start = '01234...',
                 n_random_walks_steps_back_to_ban = 0, random_walks_type = 'simple', # 'non-backtracking-beam',
                 device='Auto', dtype = 'Auto' , vec_hasher = 'Auto', verbose = 0   ):
    '''
    Random walks on Cayley/Schreier graph of permutation group with generators: "generators", 
    all starting from the same node (state): "state_rw_start". 
    Length and number of trajectories - main params.
    
    Output:
    returns X,y: X - array of states, y - number of steps rw achieves it
    X - 2d torch tensor, dimension 0 - index of state, dimension 1 - coordinates of states
    y - 1d torch tensor, dimension 0 - index of state - same as in X
    y[i] - number of rw-steps the state X[i,:] was achieved. 
    
    Input: 
    generators - generators (moves) to make random walks  (permutations), 
        can be list of vectors or array with vstacked vectors
    n_random_walk_length - number of visited nodes, i.e. number of steps + 1 
    n_random_walks_to_generate - how many random walks will run in parrallel
    state_rw_start - initial states for random walks - by default we will use 0,1,2,3 ...
        Can be vector or array
        If it is vector it will be broadcasted n_random_walks_to_generate times, 
        If it is array n_random_walks_to_generate - input n_random_walks_to_generate will be ignored
            and will be assigned: n_random_walks_to_generate = rw_start.shape[0]

    n_random_walks_steps_back_to_ban - "history depth" to store and ban revisting the previous states. (Used for non-backtracking random walks)
    '''

    if  random_walks_type == 'non-backtracking-beam':
        X,y = random_walks_nbt(state_destination=state_destination, generators=generators , 
                 n_random_walk_length = n_random_walk_length,  n_random_walks_to_generate = n_random_walks_to_generate,
                 state_rw_start = state_rw_start, # '01234...',
                 n_random_walks_steps_back_to_ban = n_random_walks_steps_back_to_ban, random_walks_type = 'non-backtracking-beam',
                 device =  device, dtype = dtype , vec_hasher = vec_hasher, verbose = verbose   )
    else:
        X,y = random_walks_simple(state_destination=state_destination, generators=generators , 
                 n_random_walk_length = n_random_walk_length,  n_random_walks_to_generate = n_random_walks_to_generate,
                 state_rw_start = state_rw_start, # '01234...',
                 device =  device, dtype = dtype ,  verbose = verbose   )

    return X,y


def random_walks_simple(state_destination, generators , n_random_walk_length,  n_random_walks_to_generate,    
                        state_rw_start = '01234...', 
                 device='Auto', dtype = 'Auto' , verbose = 0   ):

    '''
    Random walks on Cayley/Schreier graph of permutation group with generators: "generators", 
    all starting from the same node (state): "state_rw_start". 
    Length and number of trajectories - main params.
    
    Output:
    returns X,y: X - array of states, y - number of steps rw achieves it
    X - 2d torch tensor, dimension 0 - index of state, dimension 1 - coordinates of states
    y - 1d torch tensor, dimension 0 - index of state - same as in X
    y[i] - number of rw-steps the state X[i,:] was achieved. 
    
    Input: 
    generators - generators (moves) to make random walks  (permutations), 
        can be list of vectors or array with vstacked vectors

    n_random_walk_length - number of visited nodes, i.e. number of steps + 1 
    n_random_walks_to_generate - how many random walks will run in parrallel
    state_rw_start - initial states for random walks - by default we will use 0,1,2,3 ...
        Can be vector or array
        If it is vector it will be broadcasted n_random_walks_to_generate times, 
        If it is array n_random_walks_to_generate - input n_random_walks_to_generate will be ignored
            and will be assigned: n_random_walks_to_generate = rw_start.shape[0]
    '''

    ##########################################################################################
    # Analyse input params and convert to stadard forms
    ##########################################################################################

    # device 
    if device == 'Auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")    

    # Analyse input format of "generators"
    # can be list_generators, or tensor/np.array with rows - generators
    if isinstance(generators, list):
        list_generators = generators
    elif isinstance(generators, tuple):
        list_generators = list(generators)    
    elif isinstance(generators, torch.Tensor ):
        list_generators = [ list(generators[i,:]) for i in range(generators.shape[0] ) ]       
    elif isinstance(generators, np.ndarray ):
        list_generators = [list(generators[i,:]) for i in range(generators.shape[0] ) ]
    else:
        print('Unsupported format for "generators"', type(generators), generators)
        raise ValueError('Unsupported format for "generators" ' + str(type(generators)) )

    state_size = len(list_generators[0])
    all_moves = tensor_all_generators = torch.tensor( list_generators , device = device, dtype = torch.int64)
    n_generators = len(list_generators)

    if verbose >= 100:
        print('state_size', state_size )


    # dtype 
    if (state_rw_start == '01234...') or (state_rw_start == 'Auto' ):
        n_unique_symbols_in_states = state_size
    else:
        tmp = set( [int(i) for i in state_rw_start ]  ) # Number of unique elements in any iterator
        n_unique_symbols_in_states = len(tmp)
    if dtype == 'Auto':
        if n_unique_symbols_in_states <= 256:
            dtype = torch.uint8
        else:
            dtype = torch.uint16

    # Destination state
    if state_rw_start == '01234...':
        state_rw_start = torch.arange( state_size, device=device, dtype = dtype).reshape(-1,state_size)
    elif isinstance(state_destination, torch.Tensor ):
        state_rw_start =  state_destination.to(device).to(dtype).reshape(-1,state_size)
    else:
        state_rw_start = torch.tensor( state_destination, device=device, dtype = dtype).reshape(-1,state_size)

    array_of_states = state_rw_start.view(1, state_size  ).expand(n_random_walks_to_generate, state_size ).clone()


    if verbose >= 100:
        print('state_rw_start.shape:', state_rw_start.shape)
        print('state_rw_start:',state_rw_start)
        print(array_of_states.shape)
        print( array_of_states[:3,:] )


    ##########################################################################################
    # Initializations
    ##########################################################################################

    # Output: X,y - states, y - how many steps we achieve them 
    # Allocate memory: 
    X = torch.zeros( (n_random_walks_to_generate)*n_random_walk_length , state_size,    device=device, dtype = dtype )
    y = torch.zeros( (n_random_walks_to_generate)*n_random_walk_length ,    device=device, dtype = torch.uint32 )

    if verbose >= 100:
        print('X.shape',X.shape)


    # First portion of data  - just our state_rw_start state  multiplexed many times
    X[:n_random_walks_to_generate,:] = array_of_states
    y[:n_random_walks_to_generate] = 0

    # Technical to make array[ IX_array] we need  actually to write array[ range(N), IX_array  ]
    row_indices = torch.arange( array_of_states.shape[0] , device=device, dtype = dtype )
    row_indices = np.arange( array_of_states.shape[0] )[:, np.newaxis]

    # Main loop 
    for i_step in range(1,n_random_walk_length):
        y[ (i_step)*n_random_walks_to_generate : (i_step+1)*n_random_walks_to_generate  ] = i_step
        IX_moves = np.random.randint(0, n_generators, size=n_random_walks_to_generate, dtype = int) # random moves indixes
        new_array_of_states = array_of_states[ row_indices , all_moves[IX_moves,:]] # all_moves[IX_moves,:] ] 
        array_of_states = new_array_of_states
        X[ (i_step)*n_random_walks_to_generate : (i_step+1)*n_random_walks_to_generate , : ] = new_array_of_states


    if verbose >= 100:
        print(array_of_states.shape, 'array_of_states.shape')
        print(n_random_walk_length, 'n_random_walk_length', state_size,'state_size', '' )    
        print('Finished')
        print(str(X)[:500] )
        print(str(y)[:500] )

    return X,y



def random_walks_nbt(state_destination, generators , n_random_walk_length,  n_random_walks_to_generate,    state_rw_start = '01234...',
                 n_random_walks_steps_back_to_ban = 0, random_walks_type = 'non-backtracking-beam',
                 device='Auto', dtype = 'Auto' , vec_hasher = 'Auto', verbose = 0):
    '''
    Generation of improved non-backtracking random walks on Cayley/Schreier graph of permutation group with generators: "generators", 
    all starting from the same node (state): "state_rw_start". Lenght and number of trajectories - main params.
    
    Output:
    returns X,y: X - array of states, y - number of steps rw achieves it
    
    Input: 
    generators - generators (moves) to make random walks  (permutations), 
        can be list of vectors or array with vstacked vectors
    n_random_walk_length - number of visited nodes, i.e. number of steps + 1 
    n_random_walks_to_generate - how many random walks will run in parrallel
    n_random_walks_steps_back_to_ban - "history depth" to store and ban revisting the previous states. 
    state_rw_start - initial states for random walks - by default we will use 0,1,2,3 ...
        Can be vector or array
        If it is vector it will be broadcasted n_random_walks_to_generate times, 
        If it is array n_random_walks_to_generate - input n_random_walks_to_generate will be ignored
            and will be assigned: n_random_walks_to_generate = rw_start.shape[0]

    Improvements over the standard non-backtracking.
    Goal of improvements -  "mix even faster" - number of random walks steps will be better related with actual distance on graph.
    Basically the procedure is very similar to beam search, except there is no goal function.
    So we can call it "non-backtracking-beam".  
    
    1. Parameter n_random_walks_steps_back_to_ban -  "depth of history" - how many previous levels to remember and ban to visit again.
    
    2. Collective/beam baning the history - many trajectories at once are generated and we ban to visit any state from any trajectory visited before, that is each trajctory knows  the other trajectories and do not visit states visited by them
    
    3. Actually we ban non only visited states, but also "potentially visited" - that is 1-neigbours of current array of states will be banned 

    Fast implementation - fast hashes:
    The time consuming operation is the check whether the newly obtained states we already visited or not.
    To make that operation fast we employ simple, but fast hashing for integer vectors - just their scalar product with the random vector - getting single int64 number as a hash. Due to large range of int64 collisions are not observed in practice. (It can be extended to hashing with two vectors getting two int64 - then collisions would be almost impossible - but there is no practical need for that - even if collission rarely happens it will it affect overall performance.)
    Finding common/non-common elements for two int64 hash vectors - can be done effectively by e.g. torch.isin command. 
    (Other options explored here: https://stackoverflow.com/q/78896180/625396 https://stackoverflow.com/a/78634154/625396 )

    Fast implementation - applying many generators to many states/vectors at once:
    The fastest way to apply many permutations to many vectors at once is torch.gather command.
    Its interface migth not be so obvious.
    We pack it in the function get_neighbors
    See more details in the notebook: https://www.kaggle.com/code/alexandervc/permutations-numpy-torch-sympy-tutorial 
    
    Techical detail:
    Sometimes we sacrifice non-backtracking - not to crash the code.
    We generate each step n_random_walks_to_generate states, in some small examples it may happen we are not able to find states which were not visited before - then we first take as much new states as we can and add to them states which are previously visited. Such situation is not expected to happen for large groups of our interest, and the option added not to crash code on small toy examples.

    '''
    t0 = time.time()
    ##########################################################################################
    # Processing/Reformating input params
    ##########################################################################################

    # device 
    if device == 'Auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")      
    # Analyse input format of "generators"
    # can be list_generators, or tensor/np.array with rows - generators
    if isinstance(generators, list):
        list_generators = generators
    elif isinstance(generators, tuple):
        list_generators = list(generators)        
    elif isinstance(generators, torch.Tensor ):
        list_generators = [ list(generators[i,:]) for i in range(generators.shape[0] ) ]       
    elif isinstance(generators, np.ndarray ):
        list_generators = [list(generators[i,:]) for i in range(generators.shape[0] ) ]
    else:
        print('Unsupported format for "generators"', type(generators), generators)
        raise ValueError('Unsupported format for "generators" ' + str(type(generators)) )
    state_size = len(list_generators[0])
    n_generators = len( list_generators )

    # dtype 
    if (state_rw_start == '01234...') or (state_rw_start == 'Auto' ):
        n_unique_symbols_in_states = state_size
    else:
        tmp = set( [int(i) for i in state_rw_start ]  ) # Number of unique elements in any iterator
        n_unique_symbols_in_states = len(tmp)
    if dtype == 'Auto':
        if n_unique_symbols_in_states <= 256:
            dtype = torch.uint8
        else:
            dtype = torch.uint16

    # Destination state
    if (state_rw_start == '01234...') or (state_rw_start == 'Auto' ):
        state_rw_start = torch.arange( state_size, device=device, dtype = dtype).reshape(-1,state_size)
    elif isinstance(state_destination, torch.Tensor ):
        state_rw_start =  state_destination.to(device).to(dtype).reshape(-1,state_size)
    else:
        state_rw_start = torch.tensor( state_destination, device=device, dtype = dtype).reshape(-1,state_size)

    # Reformat generators in torch 2d array
    dtype_generators = torch.int64  
    tensor_generators = torch.tensor(np.array(list_generators), device=device, dtype=dtype_generators)
    #print('tensor_generators.shape:', tensor_generators.shape)

    # Preprare vector which will used for hashing - to avoid revisiting same states 
    max_int =  int( (2**62) )
    dtype_for_hash = torch.int64
    if vec_hasher == 'Auto':
        vec_hasher = torch.randint(-max_int, max_int+1, size=(state_size,), device=device, dtype=dtype_for_hash) 
        

    ##########################################################################################
    # Initializations
    ##########################################################################################

    # Main variable in the loop - store current states: 2d torch tensor
    # Initialization via state_rw_start state - duplicatie it many (n_random_walks_to_generate) times 
    array_current_states = state_rw_start.view(1, state_size  ).expand(n_random_walks_to_generate, state_size ).clone()
    
    # Output: X,y - states, y - how many steps we achieve them 
    # Allocate memory: 
    X = torch.zeros( (n_random_walks_to_generate)*n_random_walk_length , state_size, device=device, dtype = dtype )
    y = torch.zeros( (n_random_walks_to_generate)*n_random_walk_length , device=device, dtype = torch.uint32 )
    # First portion of data  - just our state_rw_start state  multiplexed many ( n_random_walks_to_generate ) times
    X[:n_random_walks_to_generate,:] = array_current_states
    y[:n_random_walks_to_generate] = 0  

    # Hash initial states. 
    if n_random_walks_steps_back_to_ban > 0:
        hash_initial_state = torch.sum( state_rw_start.view(-1, state_size  ) * vec_hasher, dim=1) # Compute hashes 
        vec_hashes_current = hash_initial_state.expand( n_random_walks_to_generate * n_generators  , n_random_walks_steps_back_to_ban ).clone()
        # That is equivalent to matmul, but matmul is not supported for ints in torch (on GPU <= 2018) - that is way round
        # Intialize index for hash storage
        # Newly obtained hash vectors will be stored in 2d array vec_hashes_current 
        # The position/column for storage: i_cyclic_index_for_hash_storage
        i_cyclic_index_for_hash_storage = 0 # Will be updated modula n_random_walks_steps_back_to_ban, i.e. from 0 to n_random_walks_steps_back_to_ban-1
        

    if verbose >= 100:
        print('X.shape:',X.shape, 'y.shape:',y.shape)
        print(array_current_states.shape)
        print( array_current_states[:3,:] )    

    ##########################################################################################
    # Main loop 
    # 1. Create new states from current making ALL possible moves - thus number of states will be more than we need
    # 2. Select those states which were not visited on "NNN" previous steps. To make it fast we uses hashes:
    # 2.1. Compute hashes of these states just by scalar multiplication on random hash vector - get single int64 as a hash
    # 2.2. Choose only those states which hashes are new - not in the stored history of hashes
    # 3. Select only desired number of states - random subset of desired size: n_random_walks_to_generate subset 
    # 4. Store these states into output X,y 
    # 5. Update hash storage 
    ##########################################################################################
    i_step_corrected = 0
    for i_step in range(1,n_random_walk_length):
        t_moves = t_hash = t_isin =  0; t_full_step = time.time() # Time profiling
        t_unique_els = 0 # not used currently 

        # 1 Create new states: 
        # Apply all generators to all current states at once
        # array_new_states: 2d array (n_random_walks_to_generate * n_generators  ) x state_size
        t1 = time.time()
        array_new_states = get_neighbors(array_current_states,tensor_generators  ).flatten(end_dim=1) # Flatten converts 3d array to 2d
        t_moves += (time.time() - t1)
        
        # 2.1 Compute hashes
        # Compute hash. For non-backtracking - selection not visited states before. 
        t1 = time.time()
        vec_hashes_new = torch.sum(array_new_states * vec_hasher, dim=1) # Compute hashes 
        # That is equivalent to matmul, but matmul is not supported for ints in torch (on GPU <= 2018) - that is way round
        t_hash += (time.time() - t1)
        #  print('hashed', t_hash )

        if n_random_walks_steps_back_to_ban > 0:
            # 2.2 Select only states not seen before
            # Nonbacktracking - select states not visited before 
            t1 = time.time()
            mask_new = ~torch.isin(vec_hashes_new, vec_hashes_current.view(-1), assume_unique=False)
            t_isin += (time.time() - t1)
            mask_new_sum = mask_new.sum().item()
            if mask_new_sum >= n_random_walks_to_generate:
                # Select only new states - not visited before
                array_new_states = array_new_states[mask_new,:]
                i_step_corrected += 1
            else:
                # Exceptional case - should not happen for large group of interest
                # The case: can not find enough new states - will take old also not to crash the code
                if mask_new_sum > 0:
                    i_tmp0 = int( np.ceil( n_random_walks_to_generate/mask_new_sum ))
                    array_new_states = array_new_states[mask_new,:].repeat(i_tmp0, 1)[:n_random_walks_to_generate,:]
                    i_step_corrected += 1
                else:
                    # do not move
                    array_new_states = array_current_states # 
                    i_step_corrected = i_step_corrected
                    
        # 3. Select only desired number of states
        # Select only n_random_walks_to_generate (with preliminary shuffling)
        # Update current states with them 
        perm = torch.randperm(array_new_states.size(0), device = device)
        array_current_states = array_new_states[perm][:n_random_walks_to_generate]

        # 4. Store results in final output
        y[ (i_step)*n_random_walks_to_generate : (i_step+1)*n_random_walks_to_generate  ] = i_step_corrected
        X[ (i_step)*n_random_walks_to_generate : (i_step+1)*n_random_walks_to_generate , : ] = array_current_states

        if n_random_walks_steps_back_to_ban>0:
            # 5. Update hash storage 
            # Pay attention - we store hashes for ALL obtained states not only for those selected - that gives us improvement:
            # We improve the chances that states obtaine on i_step will be on true graph distance i_step - our ideal goal.
            # Which might not always be the case since random walk may create loops. 
            # All the states which are achieved - they need not more than i_step steps - so it is better to ban them all
            # Thus we improve chances that the next states will increase the true graph distance
            i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1 ) % n_random_walks_steps_back_to_ban
            vec_hashes_current[:, i_cyclic_index_for_hash_storage ] = vec_hashes_new   

        if verbose >= 10:
            t_full_step = time.time()-t_full_step
            print(i_step,'i_step', 'array_current_states.shape:',array_current_states.shape, 'Time %.3f'%(time.time()-t0),
                 't_moves  %.3f, t_hash  %.3f, t_isin %.3f, t_unique_els  %.3f, t_full_step %.3f'%(t_moves , 
                  t_hash , t_isin , t_unique_els, t_full_step) )
            
    return X,y
