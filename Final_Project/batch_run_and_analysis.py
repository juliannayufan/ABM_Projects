import pandas as pd
import numpy as np
from itertools import product
from model import IndexicalityModel


def run_single_simulation(params, seed=None):
    """
    Run one simulation under a given parameter configuration.
    Returns summary statistics at the end of the run.
    """

    model = IndexicalityModel(
        N=200,
        K=params["K"],
        interaction_mode=params["interaction_mode"],
        timing_mode=params["timing_mode"],
        group_size=params.get("group_size", 5),
        batch_ratio=params.get("batch_ratio", 0.2),
        align_prob=params.get("align_prob", 0.7),
        observe_prob=params.get("observe_prob", 0.6),
        inertia=params.get("inertia", 0.3),
    )

    if seed is not None:
        np.random.seed(seed)

    max_steps = 300

    for _ in range(max_steps):
        model.step()
        if not model.running:
            break

    states = np.array([a.state for a in model.agents_list])

    stabilized = (model.running is False)

    K = params["K"]
    state_counts = {f"count_state_{k}": int(np.sum(states == k)) for k in range(K + 1)}

    return {
        "K": params["K"],
        "interaction_mode": params["interaction_mode"],
        "timing_mode": params["timing_mode"],

        # outcomes
        "final_mean": np.mean(states),
        "final_variance": np.var(states),
        "final_std": np.std(states),

        # stability
        "time_to_stability": model.time if stabilized else np.nan,
        "stable": stabilized,

        # distribution of interpretive states at equilibrium
        **state_counts
    }

def create_parameter_grid():
    """
    Full factorial design:
    interaction × timing × K
    """

    interaction_modes = ["dyadic", "group"]
    timing_modes = ["synchronous", "asynchronous", "partial"]
    K_values = [3, 5, 7, 9]

    grid = list(product(interaction_modes, timing_modes, K_values))

    param_list = []

    for interaction, timing, K in grid:
        param_list.append({
            "interaction_mode": interaction,
            "timing_mode": timing,
            "K": K,
            "group_size": 5,
            "batch_ratio": 0.2,
            "align_prob": 0.7,
            "observe_prob": 0.6,
            "inertia": 0.3,
        })

    return param_list


def run_batch(n_replicates=30):
    """
    Run multiple simulations per condition (Monte Carlo).
    """

    param_grid = create_parameter_grid()

    all_results = []

    for i, params in enumerate(param_grid):

        for r in range(n_replicates):

            seed = i * 1000 + r  # simple reproducibility scheme

            result = run_single_simulation(params, seed=seed)

            result["replicate"] = r
            all_results.append(result)

            print(
                f"Completed: {params['interaction_mode']} | "
                f"{params['timing_mode']} | K={params['K']} | rep {r}"
            )

    return pd.DataFrame(all_results)


if __name__ == "__main__":

    df = run_batch(n_replicates=30)

    df.to_csv("indexicality_batch_results.csv", index=False)

    print("\n=== BATCH RUN COMPLETE ===")
    print(df.head())