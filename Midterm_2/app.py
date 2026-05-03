import solara
from mesa.visualization import SolaraViz, make_space_component
from mesa.visualization.components import make_plot_component
from model import StandingOvationModel
from mesa.visualization.components import AgentPortrayalStyle

# Define agent portrayal
def agent_portrayal(agent):
    return AgentPortrayalStyle(
        color = "blue" if agent.standing else "red",
        marker= "s",
        size= 40,
    )

# Define spatial visualization
space = make_space_component(agent_portrayal)
# Define plot of cooperation over time
plot = make_plot_component("Standing")

# List model parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "neighborhood_type": {
        "type": "Select",
        "label": "Neighborhood",
        "value": "Five Neighbors",
        "values": ["Five Neighbors", "Cones"],
    },
    "order": {
        "type": "Select",
        "label": "Updating Rule",
        "value": "Synchronous",
        "values": ["Synchronous", "Random", "Incentive"],
    },
}


initial_model = StandingOvationModel(
    neighborhood_type="five",
    order="Synchronous",
)

page = SolaraViz(
    initial_model,
    components=[space, plot],
    model_params=model_params,
    name="Standing Ovation Problem Model"
)