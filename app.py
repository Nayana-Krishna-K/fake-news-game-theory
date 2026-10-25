
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------
st.set_page_config(
    page_title="Fake News Game Theory Simulator",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Fake News Propagation & Platform Moderation")
st.caption(
    "A Game-Theoretic Model of Fake News Propagation "
    "and Platform Moderation"
)

st.markdown("""
Explore how user sharing strategies, platform moderation costs,
and moderation effectiveness influence estimated fake-news reach.
""")

# -----------------------------------------
# MODEL ASSUMPTIONS
# -----------------------------------------
user_strategies = ["Share", "Don't Share"]
platform_strategies = ["Moderate", "Don't Moderate"]

user_payoffs = np.array([
    [2, 5],
    [1, 1]
])

platform_base_payoffs = np.array([
    [2, -3],
    [1, 3]
])

total_users = 1000
initial_reach = 10

# -----------------------------------------
# SIDEBAR: INPUT CONTROLS
# -----------------------------------------
st.sidebar.header("Simulation Settings")

moderation_cost = st.sidebar.slider(
    "Moderation cost",
    min_value=0,
    max_value=6,
    value=1
)

sharing_probability = st.sidebar.slider(
    "Sharing probability",
    min_value=0.05,
    max_value=0.80,
    value=0.30,
    step=0.05
)

moderation_effectiveness = st.sidebar.slider(
    "Moderation effectiveness (%)",
    min_value=0,
    max_value=100,
    value=80,
    step=10
) / 100

simulation_rounds = st.sidebar.slider(
    "Simulation rounds",
    min_value=1,
    max_value=20,
    value=10
)

# -----------------------------------------
# GAME-THEORY CALCULATIONS
# -----------------------------------------
platform_payoffs = platform_base_payoffs.copy()
platform_payoffs[:, 0] -= moderation_cost

user_best = (
    user_payoffs ==
    user_payoffs.max(axis=0, keepdims=True)
)

platform_best = (
    platform_payoffs ==
    platform_payoffs.max(axis=1, keepdims=True)
)

equilibrium_mask = user_best & platform_best

equilibria = []

for i, user_strategy in enumerate(user_strategies):
    for j, platform_strategy in enumerate(platform_strategies):
        if equilibrium_mask[i, j]:
            equilibria.append({
                "User Strategy": user_strategy,
                "Platform Strategy": platform_strategy,
                "User Payoff": int(user_payoffs[i, j]),
                "Platform Payoff": int(platform_payoffs[i, j])
            })

# -----------------------------------------
# PROPAGATION SIMULATION
# -----------------------------------------
def simulate_propagation(user_strategy, platform_strategy):
    reached = float(initial_reach)
    history = [reached]

    if user_strategy == "Share":
        share_probability = sharing_probability
    else:
        share_probability = 0.0

    if platform_strategy == "Moderate":
        moderation_factor = 1 - moderation_effectiveness
    else:
        moderation_factor = 1.0

    for _ in range(simulation_rounds):
        unreached = max(0, total_users - reached)

        new_reach = (
            unreached
            * share_probability
            * moderation_factor
        )

        reached = min(total_users, reached + new_reach)
        history.append(reached)

    return history

# -----------------------------------------
# TOP-LEVEL SUMMARY
# -----------------------------------------
st.subheader("Simulation Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Potential audience", f"{total_users:,}")

with col2:
    st.metric("Initial reach", initial_reach)

with col3:
    st.metric("Simulation rounds", simulation_rounds)

# -----------------------------------------
# PAYOFF MATRICES
# -----------------------------------------
st.subheader("Game-Theory Payoff Matrices")

left, right = st.columns(2)

with left:
    st.markdown("**User Payoffs**")
    st.caption("Rows = user strategies; columns = platform strategies.")
    st.dataframe(
        pd.DataFrame(
            user_payoffs,
            index=user_strategies,
            columns=platform_strategies
        ),
        use_container_width=True
    )

with right:
    st.markdown("**Platform Payoffs**")
    st.caption("Adjusted for the selected moderation cost.")
    st.dataframe(
        pd.DataFrame(
            platform_payoffs,
            index=user_strategies,
            columns=platform_strategies
        ),
        use_container_width=True
    )

# -----------------------------------------
# NASH EQUILIBRIUM
# -----------------------------------------
st.subheader("Nash Equilibrium")

if equilibria:
    st.success(f"{len(equilibria)} pure-strategy equilibrium/equilibria found.")
    st.dataframe(
        pd.DataFrame(equilibria),
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("No pure-strategy Nash equilibrium was found.")

# -----------------------------------------
# STRATEGY COMPARISON
# -----------------------------------------
st.subheader("Estimated Reach by Strategy Combination")

strategy_combinations = [
    ("Share", "Moderate"),
    ("Share", "Don't Moderate"),
    ("Don't Share", "Moderate"),
    ("Don't Share", "Don't Moderate")
]

rows = []

for user_strategy, platform_strategy in strategy_combinations:
    history = simulate_propagation(
        user_strategy,
        platform_strategy
    )

    rows.append({
        "User Strategy": user_strategy,
        "Platform Strategy": platform_strategy,
        "Estimated Reach": round(history[-1])
    })

results_df = pd.DataFrame(rows)

st.dataframe(
    results_df,
    hide_index=True,
    use_container_width=True
)

# -----------------------------------------
# GRAPHS
# -----------------------------------------
st.subheader("Propagation Visualisations")

moderated_history = simulate_propagation("Share", "Moderate")
unmoderated_history = simulate_propagation(
    "Share", "Don't Moderate"
)

graph1, graph2 = st.columns(2)

with graph1:
    fig, ax = plt.subplots(figsize=(7, 4))

    rounds = range(simulation_rounds + 1)

    ax.plot(
        rounds,
        moderated_history,
        marker="o",
        label="With moderation"
    )

    ax.plot(
        rounds,
        unmoderated_history,
        marker="s",
        label="Without moderation"
    )

    ax.set_title("Fake-News Propagation Over Time")
    ax.set_xlabel("Simulation round")
    ax.set_ylabel("Estimated cumulative reach")
    ax.set_ylim(0, total_users)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with graph2:
    fig, ax = plt.subplots(figsize=(7, 4))

    labels = (
        results_df["User Strategy"]
        + " + "
        + results_df["Platform Strategy"]
    )

    bars = ax.bar(
        labels,
        results_df["Estimated Reach"]
    )

    ax.bar_label(bars, fmt="%.0f")
    ax.set_title("Reach by Strategy Combination")
    ax.set_ylabel("Estimated users reached")
    ax.set_ylim(0, total_users)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------
# CONCLUSION AND LIMITATIONS
# -----------------------------------------
st.subheader("Interpretation")

st.markdown("""
- Sharing can increase the estimated reach of fake news.
- Moderation reduces estimated reach when moderation effectiveness is positive.
- Moderation cost changes the platform's payoffs and may change the Nash equilibrium.
- The equilibrium is determined by the payoff matrices, while propagation is estimated by a separate simplified model.
""")

st.warning("""
**Model limitations:** All payoffs and propagation parameters are illustrative
assumptions, not real-world measurements. The propagation model estimates
expected reach and does not simulate individual users or a real social network.
The results should not be interpreted as real-world predictions.
""")
