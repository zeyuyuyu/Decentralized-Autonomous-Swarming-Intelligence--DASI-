import numpy as np
from typing import List, Tuple

class SwarmAgent:
    def __init__(self, id: int, position: np.ndarray, velocity: np.ndarray):
        self.id = id
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_fitness = float('-inf')

    def update_position(self, dt: float):
        self.position += self.velocity * dt

    def update_velocity(self, global_best_position: np.ndarray, inertia: float, cognitive_weight: float, social_weight: float):
        r1 = np.random.rand(self.position.size)
        r2 = np.random.rand(self.position.size)
        self.velocity = inertia * self.velocity + \
                        cognitive_weight * r1 * (self.best_position - self.position) + \
                        social_weight * r2 * (global_best_position - self.position)

    def update_best_position(self, fitness: float):
        if fitness > self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = fitness

class SwarmIntelligence:
    def __init__(self, num_agents: int, dim: int, bounds: Tuple[np.ndarray, np.ndarray], inertia: float, cognitive_weight: float, social_weight: float):
        self.agents = [SwarmAgent(i, np.random.uniform(*bounds, size=dim), np.random.uniform(-1, 1, size=dim)) for i in range(num_agents)]
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def step(self, dt: float):
        global_best_position = np.array([agent.best_position for agent in self.agents]).mean(axis=0)
        for agent in self.agents:
            agent.update_position(dt)
            agent.update_velocity(global_best_position, self.inertia, self.cognitive_weight, self.social_weight)
            fitness = self.evaluate_fitness(agent.position)
            agent.update_best_position(fitness)

    def evaluate_fitness(self, position: np.ndarray) -> float:
        # Implement your fitness function here
        return np.linalg.norm(position)
