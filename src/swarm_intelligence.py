import numpy as np
from typing import List, Tuple, Optional

class SwarmIntelligence:
    def __init__(self, 
                 swarm_size: int = 50,
                 dimensions: int = 2,
                 cognitive_weight: float = 2.0,
                 social_weight: float = 2.0,
                 inertia_weight: float = 0.7,
                 memory_size: int = 10):
        self.swarm_size = swarm_size
        self.dimensions = dimensions
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_weight = inertia_weight
        self.memory_size = memory_size
        
        # Initialize swarm positions and velocities
        self.positions = np.random.uniform(-1, 1, (swarm_size, dimensions))
        self.velocities = np.random.uniform(-0.1, 0.1, (swarm_size, dimensions))
        
        # Personal best memory
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(swarm_size, float('inf'))
        
        # Global best memory
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        
        # Historical fitness memory for adaptation
        self.fitness_history = []
        
    def update_swarm(self, fitness_func) -> Tuple[np.ndarray, float]:
        # Evaluate current positions
        current_fitness = np.array([fitness_func(pos) for pos in self.positions])
        
        # Update personal bests
        improved_particles = current_fitness < self.pbest_fitness
        self.pbest_positions[improved_particles] = self.positions[improved_particles]
        self.pbest_fitness[improved_particles] = current_fitness[improved_particles]
        
        # Update global best
        min_fitness_idx = np.argmin(current_fitness)
        if current_fitness[min_fitness_idx] < self.gbest_fitness:
            self.gbest_fitness = current_fitness[min_fitness_idx]
            self.gbest_position = self.positions[min_fitness_idx].copy()
        
        # Store fitness history for adaptation
        self.fitness_history.append(np.mean(current_fitness))
        if len(self.fitness_history) > self.memory_size:
            self.fitness_history.pop(0)
        
        # Adapt parameters based on convergence
        self._adapt_parameters()
        
        # Update velocities and positions
        r1, r2 = np.random.random((2, self.swarm_size, self.dimensions))
        cognitive_component = self.cognitive_weight * r1 * (self.pbest_positions - self.positions)
        social_component = self.social_weight * r2 * (self.gbest_position - self.positions)
        
        self.velocities = (self.inertia_weight * self.velocities + 
                          cognitive_component + 
                          social_component)
        
        # Apply velocity clamping
        self.velocities = np.clip(self.velocities, -1, 1)
        
        # Update positions
        self.positions += self.velocities
        
        return self.gbest_position, self.gbest_fitness
    
    def _adapt_parameters(self):
        if len(self.fitness_history) < 2:
            return
            
        # Calculate improvement rate
        improvement = (self.fitness_history[-2] - self.fitness_history[-1]) / self.fitness_history[-2]
        
        # Adapt weights based on improvement
        if improvement < 0.001:  # Stagnation detected
            self.inertia_weight = min(0.9, self.inertia_weight * 1.1)
            self.cognitive_weight = min(2.5, self.cognitive_weight * 1.05)
        else:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
            self.cognitive_weight = max(1.5, self.cognitive_weight * 0.95)
    
    def reset(self):
        self.__init__(self.swarm_size, 
                     self.dimensions,
                     self.cognitive_weight,
                     self.social_weight,
                     self.inertia_weight,
                     self.memory_size)
    
    @property
    def best_solution(self) -> Optional[Tuple[np.ndarray, float]]:
        if self.gbest_position is None:
            return None
        return self.gbest_position, self.gbest_fitness