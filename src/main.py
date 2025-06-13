import argparse
from src.app import PermutationSolver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID to use')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize solver with GPU ID
    solver = PermutationSolver(config=args.config, gpu_id=args.gpu_id)
    
    # Run training
    solver.train()

if __name__ == '__main__':
    main() 