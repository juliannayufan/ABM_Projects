from mesa import Agent
import random
import numpy as np


class IndexicalAgent(Agent):
    """
    An agent representing an individual positioned within a bounded
    scalar interpretive space [0, K].

    Interpretation:
    - The agent's state is NOT an opinion or belief.
    - It represents the agent's position on an interpretive scale
      (i.e., level of interpretive resolution regarding a shared object).
    """
    def __init__(self, model):
        super().__init__(model)

        # Each agent starts at a random position in the interpretive scale
        # 0 = low interpretive resolution (coarse / minimal understanding)
        # K = high interpretive resolution (fine-grained / differentiated understanding)
        self.state = random.randint(0, model.K)

    def dyadic_align(self, other):
        """
        Dyadic interaction:
        One-to-one interaction where agents adjust their interpretive position
        based on comparison with another agent.

        Mechanism:
        - Agent compares its state with another agent.
        - Moves one step (+1 or -1) toward the other's position.
        - Movement is probabilistic (controlled by align_prob).
        - Change is also subject to inertia (resistance to change).
        """
        # Only interact with a given probability 
        # Interaction does not always lead to adjustment
        if random.random() < self.model.align_prob:
            if other.state > self.state:
                step = 1
            elif other.state < self.state:
                step = -1
            else:
                step = 0
            # Inertia introduces resistance to change (interpretive stability)
            if random.random() > self.model.inertia:
                # Ensure state stays within bounded interpretive scale [0, K]
                self.state = max(0, min(self.model.K, self.state + step))

    def group_align(self, group):
        """
        Group-based interaction:
        Agent adjusts based on the average interpretive position of a local group.

        Mechanism:
        - Agent observes multiple neighbors simultaneously.
        - Computes group mean as a proxy for local interpretive norm.
        - Moves one step toward the mean (not full convergence).
        - Movement is probabilistic and subject to inertia.
        """
        # If no neighbors exist, no update occurs
        if not group:
            return

        # Compute local interpretive norm (average state in the group)
        mean = np.mean([a.state for a in group])

        # Only update with probability (not every observation leads to change)
        if random.random() < self.model.observe_prob:
            # Move toward group interpretive norm
            if mean > self.state:
                step = 1
            elif mean < self.state:
                step = -1
            else:
                step = 0

            # Inertia: resistance to updating interpretation
            if random.random() > self.model.inertia:
                self.state = max(0, min(self.model.K, self.state + step))