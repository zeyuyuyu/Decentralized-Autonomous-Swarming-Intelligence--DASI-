import random
import networkx as nx

class SwarmIntelligence:
    def __init__(self, num_agents, communication_radius):
        self.num_agents = num_agents
        self.communication_radius = communication_radius
        self.agents = [Agent(i) for i in range(num_agents)]
        self.graph = self.create_communication_graph()

    def create_communication_graph(self):
        G = nx.Graph()
        G.add_nodes_from([agent.id for agent in self.agents])

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if self.distance(self.agents[i], self.agents[j]) <= self.communication_radius:
                    G.add_edge(self.agents[i].id, self.agents[j].id)

        return G

    def distance(self, agent1, agent2):
        return ((agent1.x - agent2.x) ** 2 + (agent1.y - agent2.y) ** 2) ** 0.5

    def update_positions(self):
        for agent in self.agents:
            agent.update_position()

    def consensus_decision(self, decision_function):
        # Distributed consensus protocol
        for i in range(10):
            for agent in self.agents:
                agent.update_state(self.graph, decision_function)
        return [agent.decision for agent in self.agents]

class Agent:
    def __init__(self, id):
        self.id = id
        self.x = random.uniform(-10, 10)
        self.y = random.uniform(-10, 10)
        self.decision = random.choice([0, 1])
        self.state = self.decision

    def update_position(self):
        self.x += random.uniform(-1, 1)
        self.y += random.uniform(-1, 1)

    def update_state(self, graph, decision_function):
        neighbors = [neighbor for neighbor in graph.neighbors(self.id)]
        self.state = decision_function(self, neighbors)
