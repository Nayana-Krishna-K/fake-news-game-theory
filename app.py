import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Game-Theoretic Fake News Model", layout="wide")

st.markdown("""
<style>
.stApp {background-color:#F7F8FA;color:#202938;font-family:"Segoe UI",Arial,sans-serif;}
h1,h2,h3 {color:#17365D;font-weight:650;letter-spacing:-0.3px;}
[data-testid="stSidebar"] {background-color:#EAF0F7;}
[data-testid="stMetric"] {background-color:#FFFFFF;border:1px solid #DCE3EC;padding:16px;border-radius:10px;}
[data-testid="stDataFrame"] {border:1px solid #DCE3EC;border-radius:8px;}
.intro {color:#596579;font-size:1.05rem;line-height:1.6;}
hr {border-color:#DCE3EC;}
</style>
""", unsafe_allow_html=True)

st.title("A Game-Theoretic Model of Fake News Propagation and Platform Moderation")
st.markdown('<p class="intro">An interactive simulation of user sharing decisions, platform moderation incentives, and estimated misinformation reach.</p>', unsafe_allow_html=True)

user_strategies = ["Share", "Don't Share"]
platform_strategies = ["Moderate", "Don't Moderate"]

# Illustrative payoffs: rows are user strategies; columns are platform strategies.
USER_PAYOFFS = np.array([[2, 5], [1, 1]], dtype=int)
PLATFORM_BASE_PAYOFFS = np.array([[2, -3], [1, 3]], dtype=int)
TOTAL_USERS = 1000
INITIAL_REACH = 10

st.sidebar.header("Simulation controls")
st.sidebar.write("Adjust the assumptions to explore the results.")

moderation_cost = st.sidebar.slider("Moderation cost", 0, 6, 1, 1,
    help="Subtracted from the platform payoff when it chooses Moderate.")
sharing_probability = st.sidebar.slider("Sharing probability", 0.05, 0.80, 0.30, 0.05,
    help="Assumed sharing probability when the user strategy is Share.")
nonsharing_factor = st.sidebar.slider("Residual sharing under 'Don't Share' (%)", 0, 30, 20, 5,
    help="Assumed fraction of the normal sharing probability that remains under Don't Share.")
moderation_effectiveness = st.sidebar.slider("Moderation effectiveness (%)", 0, 100, 80, 10) / 100.0
simulation_rounds = st.sidebar.slider("Simulation rounds", 1, 20, 10, 1)
st.sidebar.caption("All payoffs and propagation parameters are illustrative assumptions.")

def calculate_platform_payoffs(cost):
    payoffs = PLATFORM_BASE_PAYOFFS.copy()
    payoffs[:, 0] -= cost
    return payoffs

def find_nash_equilibria(cost):
    platform_payoffs = calculate_platform_payoffs(cost)
    user_best = USER_PAYOFFS == USER_PAYOFFS.max(axis=0, keepdims=True)
    platform_best = platform_payoffs == platform_payoffs.max(axis=1, keepdims=True)
    mask = user_best & platform_best
    equilibria = []
    for i, us in enumerate(user_strategies):
        for j, ps in enumerate(platform_strategies):
            if mask[i, j]:
                equilibria.append({"User Strategy": us, "Platform Strategy": ps,
                    "User Payoff": int(USER_PAYOFFS[i, j]),
                    "Platform Payoff": int(platform_payoffs[i, j])})
    return platform_payoffs, equilibria

def simulate_propagation(user_strategy, platform_strategy, share_probability,
                         residual_factor, effectiveness, rounds):
    reached = float(INITIAL_REACH)
    history = [reached]
    # "Don't Share" means a lower assumed sharing probability, not necessarily zero.
    p_share = share_probability if user_strategy == "Share" else share_probability * residual_factor
    moderation_factor = (1.0 - effectiveness) if platform_strategy == "Moderate" else 1.0
    for _ in range(rounds):
        unreached = max(0.0, TOTAL_USERS - reached)
        new_reach = unreached * p_share * moderation_factor
        reached = min(float(TOTAL_USERS), reached + new_reach)
        history.append(reached)
    return history

platform_payoffs, equilibria = find_nash_equilibria(moderation_cost)
strategy_combinations = [
    ("Share", "Moderate"),
    ("Share", "Don't Moderate"),
    ("Don't Share", "Moderate"),
    ("Don't Share", "Don't Moderate")
]
rows = []
histories = {}
for us, ps in strategy_combinations:
    hist = simulate_propagation(us, ps, sharing_probability, nonsharing_factor / 100.0,
                                moderation_effectiveness, simulation_rounds)
    histories[(us, ps)] = hist
    rows.append({"User Strategy": us, "Platform Strategy": ps,
                 "Estimated Reach": round(hist[-1])})
results_df = pd.DataFrame(rows)

st.subheader("Simulation overview")
c1, c2, c3 = st.columns(3)
