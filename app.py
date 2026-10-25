import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PAGE SETUP
st.set_page_config(
    page_title="Game-Theoretic Fake News Model",
    layout="wide"
)

# DESIGN
st.markdown("""
<style>
.stApp {
    background-color: #F7F8FA;
    color: #202938;
    font-family: "Segoe UI", Arial, sans-serif;
}
h1, h2, h3 {
    color: #17365D;
    font-weight: 650;
}
[data-testid="stSidebar"] {
    background-color: #EAF0F7;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE3EC;
    padding: 14px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.title(
    "A Game-Theoretic Model of Fake News Propagation and Platform Moderation"
)
st.write(
    "An interactive simulation of user sharing decisions, "
    "platform moderation incentives, and estimated misinformation reach."
)

# MODEL SETTINGS
USER_STRATEGIES = ["Share", "Don't Share"]
PLATFORM_STRATEGIES = ["Moderate", "Don't Moderate"]

USER_PAYOFFS = np.array([
    [2, 5],
    [1, 1]
], dtype=int)

PLATFORM_BASE_PAYOFFS = np.array([
    [2, -3],
    [1, 3]
], dtype=int)

TOTAL_USERS = 1000
INITIAL_REACH = 10

# SIDEBAR CONTROLS
st.sidebar.header("Simulation controls")
st.sidebar.write("Adjust the assumptions to explore the results.")

moderation_cost = st.sidebar.slider(
    "Moderation cost", 0, 6, 1, 1
)

sharing_probability = st.sidebar.slider(
    "Sharing probability", 0.05, 0.80, 0.30, 0.05
)

residual_percent = st.sidebar.slider(
    "Residual sharing under Don't Share (%)",
    0, 30, 20, 5
)

moderation_effectiveness = st.sidebar.slider(
    "Moderation effectiveness (%)", 0, 100, 80, 10
) / 100

simulation_rounds = st.sidebar.slider(
    "Simulation rounds", 1, 20, 10, 1
)

st.sidebar.caption(
    "All values are illustrative assumptions, not measured real-world data."
)

# PLATFORM PAYOFF MATRIX
def platform_payoff_matrix(cost):
    matrix = PLATFORM_BASE_PAYOFFS.copy()
    matrix[:, 0] -= cost
    return matrix

# NASH EQUILIBRIUM CALCULATION
def nash_equilibria(cost):
    p_payoffs = platform_payoff_matrix(cost)

    user_best_response = (
        USER_PAYOFFS ==
        USER_PAYOFFS.max(axis=0, keepdims=True)
    )

    platform_best_response = (
        p_payoffs ==
        p_payoffs.max(axis=1, keepdims=True)
    )

    mask = user_best_response & platform_best_response
    equilibria = []

    for i, user_strategy in enumerate(USER_STRATEGIES):
        for j, platform_strategy in enumerate(PLATFORM_STRATEGIES):
            if mask[i, j]:
                equilibria.append({
                    "User Strategy": user_strategy,
                    "Platform Strategy": platform_strategy,
                    "User Payoff": int(USER_PAYOFFS[i, j]),
                    "Platform Payoff": int(p_payoffs[i, j])
                })

    return p_payoffs, equilibria

# FAKE NEWS PROPAGATION SIMULATION
def simulate(user_strategy, platform_strategy):
    reached = float(INITIAL_REACH)
    history = [reached]

    if user_strategy == "Share":
        p_share = sharing_probability
    else:
        # Assumption: sharing is lower, but not necessarily zero,
        # under the Don't Share strategy.
        p_share = sharing_probability * (residual_percent / 100)

    if platform_strategy == "Moderate":
        moderation_factor = 1 - moderation_effectiveness
    else:
        moderation_factor = 1.0

    for _ in range(simulation_rounds):
        unreached = max(0.0, TOTAL_USERS - reached)

        new_reach = (
            unreached * p_share * moderation_factor
        )

        reached = min(
            float(TOTAL_USERS),
            reached + new_reach
        )

        history.append(reached)

    return history

# CALCULATE RESULTS
platform_payoffs, equilibria = nash_equilibria(moderation_cost)

combinations = [
    ("Share", "Moderate"),
    ("Share", "Don't Moderate"),
    ("Don't Share", "Moderate"),
    ("Don't Share", "Don't Moderate")
]

histories = {
    combo: simulate(*combo)
    for combo in combinations
}

results = pd.DataFrame([
    {
        "User Strategy": user_strategy,
        "Platform Strategy": platform_strategy,
        "Estimated Reach": round(histories[
            (user_strategy, platform_strategy)
        ][-1])
    }
    for user_strategy, platform_strategy in combinations
])

# SIMULATION OVERVIEW
st.subheader("Simulation overview")
st.write(
    "The model is running. The summary metrics should appear below."
)

col1, col2, col3 = st.columns(3)

col1.metric("Potential audience", f"{TOTAL_USERS:,}")
col2.metric("Initial reach", str(INITIAL_REACH))
col3.metric("Simulation rounds", str(simulation_rounds))

st.divider()

# PAYOFF MATRICES
st.subheader("Game-theory payoff matrices")

left, right = st.columns(2)

with left:
    st.markdown("**User payoff matrix**")

    user_matrix_df = pd.DataFrame(
        USER_PAYOFFS,
        index=USER_STRATEGIES,
        columns=PLATFORM_STRATEGIES
    )

    st.dataframe(user_matrix_df, use_container_width=True)

with right:
    st.markdown("**Platform payoff matrix**")

    platform_matrix_df = pd.DataFrame(
        platform_payoffs,
        index=USER_STRATEGIES,
        columns=PLATFORM_STRATEGIES
    )

    st.dataframe(platform_matrix_df, use_container_width=True)

# NASH EQUILIBRIUM
st.subheader("Nash equilibrium")

if equilibria:
    st.success(
        "Pure-strategy equilibrium(s) found for the current assumptions."
    )

    st.dataframe(
        pd.DataFrame(equilibria),
        use_container_width=True
    )
else:
    st.info(
        "No pure-strategy Nash equilibrium was found for these assumptions."
    )

# REACH TABLE
st.subheader("Estimated reach by strategy combination")

st.dataframe(results, use_container_width=True)

# GRAPHS
st.subheader("Propagation visualisations")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = list(range(simulation_rounds + 1))

    ax.plot(
        x,
        histories[("Share", "Moderate")],
        marker="o",
        linewidth=2.5,
        color="#247A68",
        label="Share + Moderate"
    )

    ax.plot(
        x,
        histories[("Share", "Don't Moderate")],
        marker="s",
        linewidth=2.5,
        color="#C45A45",
        label="Share + Don't Moderate"
    )

    ax.set_title("Estimated propagation over time")
    ax.set_xlabel("Simulation round")
    ax.set_ylabel("Estimated cumulative reach")
    ax.set_ylim(0, TOTAL_USERS)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = [
        f"{user_strategy}\n+ {platform_strategy}"
        for user_strategy, platform_strategy in combinations
    ]

    values = [
        int(histories[combo][-1])
        for combo in combinations
    ]

    bars = ax.bar(
        labels,
        values,
        color=[
            "#247A68",
            "#C45A45",
            "#6B8EAD",
            "#91A8BD"
        ]
    )

    ax.set_title("Reach by strategy combination")
    ax.set_ylabel("Estimated users reached")
    ax.set_ylim(0, TOTAL_USERS * 1.12)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# INTERPRETATION
st.subheader("Interpretation")

moderated_reach = int(
    histories[("Share", "Moderate")][-1]
)

unmoderated_reach = int(
    histories[("Share", "Don't Moderate")][-1]
)

a, b = st.columns(2)

a.metric(
    "Reach with moderation (Share)",
    moderated_reach
)

b.metric(
    "Reach without moderation (Share)",
    unmoderated_reach
)

if unmoderated_reach > 0:
    reduction = (
        (unmoderated_reach - moderated_reach)
        / unmoderated_reach
    ) * 100

    st.write(
        "Under these assumptions, moderation is associated with "
        f"an estimated reach reduction of **{reduction:.1f}%** "
        "when users choose Share."
    )

st.info(
    "The payoff matrices calculate strategic best responses and "
    "pure-strategy Nash equilibria. The propagation model separately "
    "estimates reach using the chosen assumptions."
)

# LIMITATIONS
with st.expander("Model assumptions and limitations"):
    st.markdown("""
- Payoff values are illustrative and are not calibrated to real platform data.
- Audience size, initial reach, sharing probability, residual sharing, and moderation effectiveness are assumptions.
- The Don't Share scenario uses a lower residual sharing probability. This is a modelling choice, not a measured fact.
- The propagation equation estimates cumulative reach and does not simulate individual accounts or a real social-network graph.
- Results compare scenarios; they are not real-world predictions or causal evidence.
- Future work could use a real network structure, varied user behaviours, random trials, and empirically justified parameters.
""")

st.caption(
    "Academic prototype · Simulated results only · Not real-world measurements"
)
