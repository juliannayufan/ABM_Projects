from mesa import Model
from mesa.space import SingleGrid, MultiGrid
from mesa.datacollection import DataCollector
import random
import numpy as np
from agents import IndexicalAgent


class IndexicalityModel(Model):
    """
    This model simulates how interpretations of a shared
    object evolve through local interaction.

    Core idea:
    - Agents do NOT learn truth or gain ability
    - Agents shift their *situated interpretive position*
      within a bounded space defined by K

    Implements:
    (1) interaction_mode → dyadic vs group
    (2) timing_mode → synchronous / asynchronous / partial
    (3) K → interpretive resolution
    """
    def __init__(
        self,
        width=20,
        height=20,
        seed=None,
        N=200, # total number of agents
        K=5,
        interaction_mode="dyadic", # dyadic (default) vs group
        timing_mode="asynchronous",  # synchronous / asynchronous (default) / partial
        group_size=5, # number of neighbors used in group interaction
        batch_ratio=0.2,  # for partial sync
        align_prob=0.7, # probability that interaction produces change
        observe_prob=0.6,
        inertia=0.3,
        stability_threshold=0.02,
        window=10
        ):

        if seed is not None:
            seed = int(seed)
        super().__init__(rng=seed)

        self.width = width
        self.height = height
        self.N = N
        # K = global bounded interpretive resolution
        self.K = K

        self.interaction_mode = interaction_mode
        self.timing_mode = timing_mode
        self.group_size = group_size
        self.batch_ratio = batch_ratio

        self.align_prob = align_prob
        self.observe_prob = observe_prob
        self.inertia = inertia

        self.stability_threshold = stability_threshold
        self.window = window

        # ---------------- GRID INITIALIZATION (STRICT SINGLE OCCUPANCY) ----------------
        cells = width * height

        if N > cells:
            raise ValueError(
                f"N={N} exceeds grid capacity ({cells}). "
                "This model requires exactly one agent per cell."
            )

        self.grid = SingleGrid(width, height, torus=True)

        # unique positions for all agents (fixed spatial topology)
        positions = [(x, y) for x in range(width) for y in range(height)]
        random.shuffle(positions)

        # ---------------- AGENT CREATION ----------------
        self.agents_list = []
        self.history = []
        self.running = True
        self.time = 0

        for i in range(N):
            a = IndexicalAgent(self)
            self.agents_list.append(a)

            # guaranteed unique cell
            pos = positions.pop()
            self.grid.place_agent(a, pos)

        # Data collector for tracking mean state over time
        self.datacollector = DataCollector(
            model_reporters={
                "Mean State": lambda m: np.mean([ag.state for ag in m.agents_list])
            }
        )

        self.datacollector.collect(self)

    def is_stable(self):
        if len(self.history) < self.window:
            return False
        return np.std(self.history[-self.window:]) < self.stability_threshold

    # define step as function of activation regime
    def step(self):
        agents = self.agents_list[:]

        # All agents decide first, then updates are applied simultaneously
        if self.timing_mode == "synchronous":
            updates = []
            for a in agents:
                neighbors = self.grid.get_neighbors(a.pos, moore=True, include_center=False)
                # one interaction partner
                if self.interaction_mode == "dyadic":
                    if neighbors:
                        updates.append((a, random.choice(neighbors)))
                else:
                    # group interaction
                    if neighbors:
                        updates.append((a, random.sample(neighbors, min(self.group_size, len(neighbors)))))

            # apply all at once
            for item in updates:
                a = item[0]
                partner_or_group = item[1]

                if self.interaction_mode == "dyadic":
                    a.dyadic_align(partner_or_group)
                else:
                    a.group_align(partner_or_group)

        # Agents update sequentially in randomized order
        elif self.timing_mode == "asynchronous":

            random.shuffle(agents)

            for a in agents:
                neighbors = self.grid.get_neighbors(a.pos, moore=True, include_center=False)

                if self.interaction_mode == "dyadic":
                    if neighbors:
                        a.dyadic_align(random.choice(neighbors))
                else:
                    if neighbors:
                        a.group_align(random.sample(neighbors, min(self.group_size, len(neighbors))))

        # Population is divided into batches; each batch updates sequentially
        elif self.timing_mode == "partial":
            random.shuffle(agents)
            # size of each update batch
            batch_size = max(2, int(len(agents) * self.batch_ratio))
            batches = [agents[i:i + batch_size] for i in range(0, len(agents), batch_size)]

            for batch in batches:
                batch_updates = []
                # collect all updates in batch first
                for a in batch:
                    neighbors = self.grid.get_neighbors(a.pos, moore=True, include_center=False)

                    if not neighbors:
                        continue

                    if self.interaction_mode == "dyadic":
                        batch_updates.append((a, random.choice(neighbors)))
                   
                    else:
                            batch_updates.append(
                                (a, random.sample(neighbors, min(self.group_size, len(neighbors))))
                            )
                # then apply batch updates
                for a, partner_or_group in batch_updates:
                    if self.interaction_mode == "dyadic":
                        a.dyadic_align(partner_or_group)
                    else:
                        a.group_align(partner_or_group)
        else:
            raise ValueError("Unknown timing_mode")

        # tracking
        mean_state = np.mean([a.state for a in agents])
        self.history.append(mean_state)
        self.time += 1
        
        # stop simulation if convergence detected
        if self.is_stable():
            self.running = False
        
        self.datacollector.collect(self)