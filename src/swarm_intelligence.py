# Swarm Intelligence Implementation with Particle Swarm Optimization
import numpy as np
from typing import List, Tuple, Callable

class Particle:
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]]):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros(dimensions)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

class SwarmIntelligence:
    def __init__(self, n_particles: int, dimensions: int, bounds: List[Tuple[float, float]]):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.particles = [Particle(dimensions, bounds) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_score = float('inf')
        
        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 2.0  # Cognitive parameter
        self.c2 = 2.0  # Social parameter

    def optimize(self, objective_func: Callable, max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """Optimize using Particle Swarm Optimization algorithm"""
        for _ in range(max_iterations):
            for particle in self.particles:
                # Evaluate current position
                score = objective_func(particle.position)
                
                # Update particle's best
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
                
                # Update velocity and position
                r1, r2 = np.random.rand(2)
                particle.velocity = (self.w * particle.velocity +
                                   self.c1 * r1 * (particle.best_position - particle.position) +
                                   self.c2 * r2 * (self.global_best_position - particle.position))
                
                # Update position with bounds checking
                particle.position += particle.velocity
                for i, (low, high) in enumerate(self.bounds):
                    particle.position[i] = np.clip(particle.position[i], low, high)
        
        return self.global_best_position, self.global_best_score

    def adapt_parameters(self, iteration: int, max_iterations: int):
        """Adapt PSO parameters based on iteration progress"""
        # Linearly decrease inertia weight
        self.w = 0.9 - (0.9 - 0.4) * (iteration / max_iterations)
        
        # Adapt cognitive and social parameters
        self.c1 = 2.5 - 2 * (iteration / max_iterations)
        self.c2 = 0.5 + 2 * (iteration / max_iterations)

    def get_swarm_diversity(self) -> float:
        """Calculate swarm diversity as mean pairwise distance"""
        positions = np.array([p.position for p in self.particles])
        distances = np.sqrt(((positions[:, np.newaxis] - positions) ** 2).sum(axis=2))
        return np.mean(distances)

    def reset_swarm(self):
        """Reset swarm to initial random positions"""
        self.particles = [Particle(self.dimensions, self.bounds) for _ in range(self.n_particles)]
        self.global_best_score = float('inf')
        self.global_best_position = None