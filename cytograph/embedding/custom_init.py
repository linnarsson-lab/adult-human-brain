import numpy as np
import loompy
from openTSNE import initialization
from cytograph.embedding import art_of_tsne

def pull_towards_levels(X: np.ndarray, pull_levels: np.ndarray, \
                        min_force: float = 0.6, max_force: float = 0.95) -> np.ndarray:
    """
        Alternative initialization method for art_of_tsne that pulls points from an initial pca layout
        towards a radius proportional to their relative pull_level. The force is random but limited by min_force - max_force
        Typically, the pull_level may be the pseudoage to force early stages close to center.
        Args:
            X             The data matrix of shape (n_cells, n_genes) i.e. (n_samples, n_features)
            pull_levels   The metric that determines "optimal" radius for each point
            min_force     Minimum force to apply
            max_force     Maximum force to apply
        
        Returns:
                The adjusted initial embedding

    """
    initial_embedding = initialization.pca(X)
    ncells = initial_embedding.shape[0]
    if ncells != len(pull_levels):
        print("pull_levels has to be equal size as number of cells!")
    min_pull_level, max_pull_level = min(pull_levels), max(pull_levels)
    pull_range = max_pull_level - min_pull_level
    curr_radius = np.linalg.norm(initial_embedding, axis=1) # Get distances to center using zero array
    move_force = np.random.uniform(min_force, max_force, ncells)
    target_radius = (pull_levels - min_pull_level) / pull_range * 0.5
    adjust_factor = (move_force * target_radius + (1.0 - move_force) * curr_radius) / curr_radius
    adjusted_initial_embedding = initial_embedding * adjust_factor[:, np.newaxis]
    return adjusted_initial_embedding

