from mesa import Agent
import random

class SchellingAgent(Agent):
    ## Initiate agent instance, inherit model trait from parent class
    def __init__(self, model, agent_type):
        super().__init__(model)
        ## Set agent type
        self.type = agent_type

        # Set agent initial tolerance, a random value between 0.3 and 0.5.
        self.tolerance = random.uniform(0.3, 0.5)

        # Add learning attributes
        # Track agent's tolerance performance
        self.score = 0
        self.next_tolerance = self.tolerance

    # Define a step that contains both learn and move functions.
    def step(self):
        self.learn()
        self.move()

    # Add learning from the best neighbor into decision rule
    def learn(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.model.radius, include_center=False
        )
        neighbors_plus_me = neighbors + [self]

        best_neighbor = max(neighbors_plus_me, key=lambda a: a.score)
        self.next_tolerance = best_neighbor.tolerance

        # Update tolerance before moving
        self.tolerance = self.next_tolerance

    ## Define basic decision rule
    def move(self):
        ## Get list of neighbors within range of sight
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore = True, radius = self.model.radius, include_center = False)
        ## Count neighbors of same type as self
        similar_neighbors = len([n for n in neighbors if n.type == self.type])
        ## If an agent has any neighbors (to avoid division by zero), calculate share of neighbors of same type
        if (valid_neighbors := len(neighbors)) > 0:
            share_alike = similar_neighbors / valid_neighbors
        else:
            share_alike = 0
        ## If unhappy with neighbors, move to random empty slot. Otherwise add one to model count of happy agents.
        # Here, changes have made, "share_alike" compares with "self.tolerance" of each agent to reflect learned tolerance
        if share_alike < self.tolerance:
            self.model.grid.move_to_empty(self)
        else: 
            self.model.happy +=1   
            # Keep track of tolerance performance
            self.score += 1
