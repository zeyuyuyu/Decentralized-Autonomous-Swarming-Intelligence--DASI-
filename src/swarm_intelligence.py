import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class AgentRole(Enum):
    EXPLORER = 'explorer'
    WORKER = 'worker'
    COORDINATOR = 'coordinator'

@dataclass
class Agent:
    id: int
    position: Tuple[float, float]
    role: AgentRole
    energy: float = 100.0
    memory: Dict = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = {}

class SwarmIntelligence:
    def __init__(self, num_agents: int, world_size: Tuple[float, float]):
        self.world_size = world_size
        self.agents = self._initialize_agents(num_agents)
        self.pheromone_map = np.zeros(world_size)
        self.learning_rate = 0.1
        
    def _initialize_agents(self, num_agents: int) -> List[Agent]:
        agents = []
        roles = list(AgentRole)
        for i in range(num_agents):
            pos = (np.random.uniform(0, self.world_size[0]),
                  np.random.uniform(0, self.world_size[1]))
            role = roles[i % len(roles)]
            agents.append(Agent(id=i, position=pos, role=role))
        return agents

    def update_roles(self) -> None:
        """Dynamically update agent roles based on swarm needs"""
        explorer_count = sum(1 for agent in self.agents if agent.role == AgentRole.EXPLORER)
        worker_count = sum(1 for agent in self.agents if agent.role == AgentRole.WORKER)
        
        for agent in self.agents:
            # Adapt roles based on current situation
            if agent.energy < 30 and agent.role == AgentRole.EXPLORER:
                agent.role = AgentRole.WORKER
            elif agent.energy > 80 and worker_count > len(self.agents) // 2:
                agent.role = AgentRole.EXPLORER

    def move_agents(self) -> None:
        """Update agent positions using swarm behavior rules"""
        for agent in self.agents:
            # Get neighboring agents
            neighbors = self._get_neighbors(agent)
            
            if agent.role == AgentRole.EXPLORER:
                self._explore(agent, neighbors)
            elif agent.role == AgentRole.WORKER:
                self._work(agent, neighbors)
            elif agent.role == AgentRole.COORDINATOR:
                self._coordinate(agent, neighbors)

            # Update pheromone trails
            self._update_pheromones(agent)

    def _get_neighbors(self, agent: Agent, radius: float = 10.0) -> List[Agent]:
        """Find neighboring agents within radius"""
        return [
            other for other in self.agents
            if other.id != agent.id
            and np.linalg.norm(np.array(agent.position) - np.array(other.position)) < radius
        ]

    def _explore(self, agent: Agent, neighbors: List[Agent]) -> None:
        """Implement explorer behavior"""
        # Add random movement with collision avoidance
        movement = np.random.uniform(-1, 1, 2)
        if neighbors:
            separation = self._calculate_separation(agent, neighbors)
            movement += separation
        
        new_pos = np.array(agent.position) + movement
        agent.position = tuple(np.clip(new_pos, 0, self.world_size[0]))
        agent.energy -= 0.5

    def _work(self, agent: Agent, neighbors: List[Agent]) -> None:
        """Implement worker behavior"""
        # Follow pheromone trails and perform tasks
        gradient = self._get_pheromone_gradient(agent.position)
        movement = gradient * self.learning_rate
        
        if neighbors:
            cohesion = self._calculate_cohesion(agent, neighbors)
            movement += cohesion
            
        new_pos = np.array(agent.position) + movement
        agent.position = tuple(np.clip(new_pos, 0, self.world_size[0]))
        agent.energy -= 0.3

    def _coordinate(self, agent: Agent, neighbors: List[Agent]) -> None:
        """Implement coordinator behavior"""
        # Maintain optimal position relative to other agents
        if neighbors:
            center = np.mean([n.position for n in neighbors], axis=0)
            movement = (center - np.array(agent.position)) * 0.1
            
            new_pos = np.array(agent.position) + movement
            agent.position = tuple(np.clip(new_pos, 0, self.world_size[0]))
        agent.energy -= 0.1

    def _calculate_separation(self, agent: Agent, neighbors: List[Agent]) -> np.ndarray:
        """Calculate separation vector to avoid collisions"""
        if not neighbors:
            return np.zeros(2)
        separation = np.zeros(2)
        for neighbor in neighbors:
            diff = np.array(agent.position) - np.array(neighbor.position)
            distance = np.linalg.norm(diff)
            if distance < 5.0:  # Minimum separation distance
                separation += diff / (distance ** 2)
        return separation

    def _calculate_cohesion(self, agent: Agent, neighbors: List[Agent]) -> np.ndarray:
        """Calculate cohesion vector to maintain swarm unity"""
        if not neighbors:
            return np.zeros(2)
        center = np.mean([n.position for n in neighbors], axis=0)
        return (center - np.array(agent.position)) * 0.1

    def _update_pheromones(self, agent: Agent) -> None:
        """Update pheromone trails based on agent activity"""
        x, y = int(agent.position[0]), int(agent.position[1])
        if 0 <= x < self.world_size[0] and 0 <= y < self.world_size[1]:
            self.pheromone_map[x, y] += 1.0
        # Pheromone evaporation
        self.pheromone_map *= 0.95

    def _get_pheromone_gradient(self, position: Tuple[float, float]) -> np.ndarray:
        """Calculate pheromone gradient at given position"""
        x, y = int(position[0]), int(position[1])
        if 0 < x < self.world_size[0]-1 and 0 < y < self.world_size[1]-1:
            dx = self.pheromone_map[x+1, y] - self.pheromone_map[x-1, y]
            dy = self.pheromone_map[x, y+1] - self.pheromone_map[x, y-1]
            return np.array([dx, dy])
        return np.zeros(2)

    def step(self) -> None:
        """Perform one step of swarm simulation"""
        self.update_roles()
        self.move_agents()
        # Replenish energy for agents near resources
        for agent in self.agents:
            if agent.energy < 100:
                agent.energy = min(100, agent.energy + 0.1)