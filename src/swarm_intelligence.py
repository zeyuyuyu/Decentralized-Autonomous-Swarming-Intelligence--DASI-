import numpy as np
from typing import List, Tuple, Callable

class AdaptiveParticleSwarm:
    def __init__(self, n_particles: int, dimensions: int, 
                 fitness_func: Callable[[np.ndarray], float],
                 w_min: float = 0.4, w_max: float = 0.9,
                 c1_start: float = 2.5, c1_end: float = 0.5,
                 c2_start: float = 0.5, c2_end: float = 2.5):
        
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.fitness_func = fitness_func
        
        # Inertia weight bounds
        self.w_min = w_min
        self.w_max = w_max
        
        # Acceleration coefficients bounds
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end
        
        # Initialize particles
        self.positions = np.random.uniform(-1, 1, (n_particles, dimensions))
        self.velocities = np.random.uniform(-0.1, 0.1, (n_particles, dimensions))
        
        # Personal best
        self.pbest_pos = self.positions.copy()
        self.pbest_scores = np.array([self.fitness_func(p) for p in self.positions])
        
        # Global best
        self.gbest_idx = np.argmin(self.pbest_scores)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_score = self.pbest_scores[self.gbest_idx]
        
        # Dynamic topology matrix
        self.topology = np.ones((n_particles, n_particles))
        
    def update_topology(self, iteration: int, max_iter: int):
        """Updates the neighborhood topology based on particle performance"""
        # Calculate particle diversity
        diversity = np.std(self.positions, axis=0).mean()
        
        # Adjust connectivity based on optimization phase
        if iteration < max_iter * 0.4:  # Exploration phase
            k = max(2, int(self.n_particles * 0.2))  # Sparse connectivity
        else:  # Exploitation phase
            k = max(5, int(self.n_particles * 0.5))  # Dense connectivity
            
        # Update topology using k-nearest neighbors
        for i in range(self.n_particles):
            distances = np.linalg.norm(self.positions - self.positions[i], axis=1)
            nearest = np.argsort(distances)[:k]
            self.topology[i] = 0
            self.topology[i, nearest] = 1
    
    def update_parameters(self, iteration: int, max_iter: int):
        """Updates inertia weight and acceleration coefficients"""
        progress = iteration / max_iter
        
        # Update inertia weight
        self.w = self.w_max - (self.w_max - self.w_min) * progress
        
        # Update acceleration coefficients
        self.c1 = self.c1_start - (self.c1_start - self.c1_end) * progress
        self.c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
    
    def optimize(self, max_iter: int, bounds: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, float]:
        """Main optimization loop"""
        lower_bound, upper_bound = bounds
        
        for iteration in range(max_iter):
            # Update swarm parameters
            self.update_parameters(iteration, max_iter)
            self.update_topology(iteration, max_iter)
            
            # Update particles
            for i in range(self.n_particles):
                # Get neighborhood best
                neighbors = np.where(self.topology[i] == 1)[0]
                neighbor_best_idx = neighbors[np.argmin(self.pbest_scores[neighbors])]
                lbest_pos = self.pbest_pos[neighbor_best_idx]
                
                # Update velocity
                r1, r2 = np.random.rand(2)
                cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
                social = self.c2 * r2 * (lbest_pos - self.positions[i])
                self.velocities[i] = (self.w * self.velocities[i] + cognitive + social)
                
                # Update position
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lower_bound, upper_bound)
                
                # Update personal best
                score = self.fitness_func(self.positions[i])
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_pos[i] = self.positions[i].copy()
                    
                    # Update global best
                    if score < self.gbest_score:
                        self.gbest_score = score
                        self.gbest_pos = self.positions[i].copy()
            
        return self.gbest_pos, self.gbest_score

def create_swarm(n_particles: int, dimensions: int, 
                fitness_func: Callable[[np.ndarray], float]) -> AdaptiveParticleSwarm:
    """Factory function to create a new swarm instance"""
    return AdaptiveParticleSwarm(n_particles, dimensions, fitness_func)