from mesa import Agent
import random


class AudienceAgent(Agent):
    """
    Each individual has an identical standing threshold of 0.5, 
    and thus she will stand initially if and only if her perceived
    quality exceeds 0.5.

    - Initial state: stand if q_ij >= 0.5
    - Tie-breaking randomness
    - Neighborhoods: five vs. cone

    """
    # Initialize agent
    def __init__(self, unique_id, model, q):
        super().__init__(model)
        self.unique_id = unique_id
        self.q = q
        self.threshold = 0.5 # identical standing threshold of 0.5
        self.standing = self.q >= self.threshold
        self.next_state = self.standing
        

    # Neighborhood selection: five or cone
    def get_neighbors(self):
        if self.model.neighborhood_type == "five":
            return self.get_five_neighbors()
        return self.get_cone_neighbors()

    # Prevent agents access positions outside the grid
    def valid_agents(self, positions):
        agents = []
        for pos in positions:
            if not self.model.grid.out_of_bounds(pos):
                cell = self.model.grid.get_cell_list_contents([pos])
                # in case of future change of seating capacity, 
                # although right now every position has an agent
                if cell: 
                    agents.append(cell[0])
        return agents

    def get_five_neighbors(self):
        x, y = self.pos
        positions = [
            (x - 1, y), (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)
        ]
        return self.valid_agents(positions)

    def get_cone_neighbors(self):
        x, y = self.pos
        positions = []
        
        positions.append((x - 1, y))
        positions.append((x + 1, y))
        
        # stop counting at "7 agents 3 rows ahead" 
        # according to the research visualization
        for d in range(1, 4): # only 3 rows
            for dx in range(-d, d + 1):
                positions.append((x + dx, y + d))
        return self.valid_agents(positions)

    # Decison rules
    # Each audience member uses a majority rule heuristic—if a majority
    # of the people that she sees are standing, she stands, if not she sits. 
    def pick_move(self):
        neighbors = self.get_neighbors()
        if not neighbors:
            self.next_state = self.standing
            return

        stand = sum(n.standing for n in neighbors)
        sit = len(neighbors) - stand

        if stand > sit:
            self.next_state = True
        elif sit > stand:
            self.next_state = False
        # tiebreak: "In the case of a tie, 
        # we assume that she sits or stands with equal probability."
        else:
            self.next_state = random.choice([True, False]) 
    
    def update(self):
        self.standing = self.next_state

    # Used for incentive-based updating
    def mismatch_with_neighbors(self):
        neighbors = self.get_neighbors()
        if not neighbors:
            return 0

        stand = sum(n.standing for n in neighbors)
        sit = len(neighbors) - stand
        majority = stand > sit
        return abs((1 if self.standing else 0) - majority)