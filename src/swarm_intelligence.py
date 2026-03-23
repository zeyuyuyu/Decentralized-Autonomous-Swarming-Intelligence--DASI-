import numpy as np
from typing import List, Tuple, Optional

class SwarmParticle:
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]]):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros(dimensions)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

class AdaptiveSwarmIntelligence:
    def __init__(self, 
                 n_particles: int,
                 dimensions: int,
                 bounds: List[Tuple[float, float]],
                 inertia_weight: float = 0.7,
                 cognitive_weight: float = 1.5,
                 social_weight: float = 1.5,
                 min_velocity: float = -1.0,
                 max_velocity: float = 1.0):
        
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.w = inertia_weight
        self.c1 = cognitive_weight
        self.c2 = social_weight
        self.velocity_bounds = (min_velocity, max_velocity)
        
        self.particles = [SwarmParticle(dimensions, bounds) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.iteration = 0
        
    def evaluate(self, objective_func) -> None:
        for particle in self.particles:
            score = objective_func(particle.position)
            
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()

    def update_velocities(self) -> None:
        for particle in self.particles:
            r1, r2 = np.random.rand(2)
            
            cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
            social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
            
            # Update velocity with inertia and momentum
            particle.velocity = (self.w * particle.velocity + 
                               cognitive_velocity + 
                               social_velocity)
            
            # Apply velocity clamping
            particle.velocity = np.clip(particle.velocity, 
                                      self.velocity_bounds[0],
                                      self.velocity_bounds[1])

    def update_positions(self) -> None:
        for particle in self.particles:
            particle.position += particle.velocity
            
            # Enforce position bounds
            for i, (low, high) in enumerate(self.bounds):
                particle.position[i] = np.clip(particle.position[i], low, high)

    def optimize(self, objective_func, max_iterations: int, target_score: Optional[float] = None) -> Tuple[np.ndarray, float]:
        for _ in range(max_iterations):
            self.evaluate(objective_func)
            
            if target_score is not None and self.global_best_score <= target_score:
                break
                
            self.update_velocities()
            self.update_positions()
            self.iteration += 1
            
            # Adaptive inertia weight
            self.w = max(0.4, self.w * 0.99)
            
        return self.global_best_position, self.global_best_score

    def get_swarm_state(self) -> dict:
        return {
            'iteration': self.iteration,
            'global_best_score': self.global_best_score,
            'global_best_position': self.global_best_position,
            'current_inertia': self.w,
            'particle_positions': [p.position for p in self.particles],
            'particle_velocities': [p.velocity for p in self.particles]
        }