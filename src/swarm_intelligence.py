import numpy as np

class SwarmCoordination:
    def __init__(self, num_agents, num_tasks, task_locations, agent_locations):
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.task_locations = task_locations
        self.agent_locations = agent_locations
        self.task_assignments = [[] for _ in range(self.num_agents)]

    def assign_tasks(self):
        """Assign tasks to agents using a swarm coordination algorithm."""
        # Calculate the distance between each agent and each task
        distances = np.zeros((self.num_agents, self.num_tasks))
        for i in range(self.num_agents):
            for j in range(self.num_tasks):
                distances[i, j] = np.linalg.norm(self.agent_locations[i] - self.task_locations[j])

        # Assign tasks to agents using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distances)

        # Update the task assignments for each agent
        for i, j in zip(row_ind, col_ind):
            self.task_assignments[i].append(j)

        return self.task_assignments

def linear_sum_assignment(cost_matrix):
    """Solve the linear sum assignment problem using the Hungarian algorithm."""
    # Implementation of the Hungarian algorithm
    # ...
    return row_ind, col_ind
