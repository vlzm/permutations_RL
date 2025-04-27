# Permutations RL

Machine Learning methods for group theory and Cayley graphs pathfinding.

## Project Structure

```
permutations_RL/
├── config/           # Configuration files
├── src/             # Source code
│   ├── models/      # Neural network models (MLP, DQN)
│   ├── data/        # Data generation and processing
│   ├── search/      # Search algorithms
│   └── utils/       # Utility functions
├── notebooks/       # Jupyter notebooks
└── tests/          # Unit tests
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main experiments are located in the notebooks directory.

## References

- Paper 1: https://arxiv.org/abs/2502.18663
- Paper 2: https://arxiv.org/abs/2502.13266 