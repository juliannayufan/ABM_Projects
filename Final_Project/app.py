import solara
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from mesa.visualization import SolaraViz, make_space_component
from mesa.visualization.components import make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from model import IndexicalityModel


# =========================================================
# 0. COLOR MAP (GLOBAL, NOT RECOMPUTED PER AGENT)
# =========================================================
cmap = cm.get_cmap("viridis")


def get_color(value, K):
    """
    Map discrete state (0..K) onto a continuous color scale.

    Interpretation:
    - state = position in interpretive resolution space
    - K = global scale constraint
    """
    return cmap(value / max(K, 1))


# Define agent portrayal
def agent_portrayal(agent):
    K = agent.model.K
    state = int(agent.state)

    # ensure bounds safety
    state = max(0, min(state, K))

    return AgentPortrayalStyle(
        # colors representing interpretive positions
        color=mcolors.to_hex(get_color(state, K)),
        marker="o",
        size=12 + (state / max(K, 1)) * 10,
    )


# Define spatial visualization
space = make_space_component(agent_portrayal)

# # Define plot of cooperation over time
plot = make_plot_component("Mean State")

# List model parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },

    "K": {
        "type": "SliderInt",
        "value": 5,
        "min": 3,
        "max": 11,
        "label": "Scale (K)",
    },

    "interaction_mode": {
        "type": "Select",
        "label": "Interaction Mode",
        "value": "dyadic",
        "values": ["dyadic", "group"],
    },

    "timing_mode": {
        "type": "Select",
        "label": "Timing Regime",
        "value": "asynchronous",
        "values": ["asynchronous", "synchronous", "partial"],
    },

    "group_size": {
        "type": "SliderInt",
        "value": 5,
        "min": 2,
        "max": 10,
        "label": "Group Size",
    },

    "batch_ratio": {
        "type": "SliderFloat",
        "value": 0.2,
        "min": 0,
        "max": 0.5,
        "step": 0.01,
        "label": "Partial Sync Ratio",
    },

    "align_prob": {
        "type": "SliderFloat",
        "value": 0.7,
        "min": 0,
        "max": 1,
        "step": 0.01,
        "label": "Alignment Probability",
    },

    "observe_prob": {
        "type": "SliderFloat",
        "value": 0.6,
        "min": 0,
        "max": 1,
        "step": 0.01,
        "label": "Observation Probability",
    },

    "inertia": {
        "type": "SliderFloat",
        "value": 0.3,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "label": "Inertia",
    },
}


initial_model = IndexicalityModel(
    K=5,
    interaction_mode="dyadic",
    timing_mode="asynchronous",
)

# annotations
legend = solara.Card(
    title="Model Visualization",
    children=[
        solara.Markdown("""
### Implements:
- interaction_mode → dyadic vs group
- timing_mode → synchronous / asynchronous / partial
-  K → interpretive resolution: each agent has a state in [0, K]

---

### What different colors stand for:
- Dark = low interpretive resolution (e.g., purple, dark green)
- Bright = high interpretive resolution (e.g., light green, yellow)

---

### Interpretation
Colors of agents are NOT their intelligibility of an object/topic.
They are situated metapragmatic opinions in a bounded interpretive scale.
        """)
    ],
)


# layout
@solara.component
def Page():
    solara.Column([
        legend,
        SolaraViz(
            initial_model,
            components=[space, plot],
            model_params=model_params,
            name="Interpretive Scale Dynamics"
        )
    ])


page = Page

