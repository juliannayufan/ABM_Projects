from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from agents import AudienceAgent
import random


class StandingOvationModel(Model):
    """
    Implements:
    - Information Structure: Five vs Cone neighborhoods
    - Timing of Updating: Synchronous / Async-Random / Async-Incentive updating
    """

    def __init__(
        self, 
        width=20, 
        height=20, 
        neighborhood_type="five", 
        order="Synchronous", 
        seed=None
    ):
        if seed is not None:
            seed = int(seed)
        super().__init__(rng=seed)

        self.order = order
        self.neighborhood_type = neighborhood_type

        self.grid = SingleGrid(width, height, torus=False) # fixed seats and hard boundaries

        # Seats all filled with agents, one per seats
        for x in range(width):
            for y in range(height):
                q = random.uniform(0, 1)  #initial quality assessment in the interval[0, 1]
                agent = AudienceAgent((x, y), self, q)
                self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "Standing": lambda m: sum(a.standing for a in m.agents)
            }
        )

    # define step as function of activation regime
    def step(self):
        # Synchronous updating: all agents update in unison
        if self.order == "Synchronous":
            # all agents decide simultaneously
            self.agents.do("pick_move")
            # all agents update simultaneously
            self.agents.do("update")
        
        # Asynchronous-Random updating: agents permuted into random order and updated one at a time
        elif self.order == "Random":
            # shuffle agents and call pick_move and update on each
            agents = list(self.agents)
            self.random.shuffle(agents)
            for a in agents:
                a.pick_move()
                a.update()
        
        # Asynchronous-Incentive-Based updating: agents update one at a time based on ordering rule
        # agents least like their neighbors move first
        elif self.order == "Incentive":
            # sort agents by mismatch with neighbors (highest mismatch first)
            sorted_agents = sorted(
                self.agents,
                key=lambda a: a.mismatch_with_neighbors(),
                reverse=True
            )
            # move and update each agent in sorted order
            for a in sorted_agents:
                a.pick_move()
                a.update()

        else:
            raise ValueError(f"Unknown order: {self.order}")

        self.datacollector.collect(self)
