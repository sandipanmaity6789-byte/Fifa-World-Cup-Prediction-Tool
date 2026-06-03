import streamlit as st
import pandas as pd
import numpy as np
import pickle
with open('world_cup_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
df = pd.read_csv('streamlit_teams.csv')
team_options = sorted(list(df['team'].values))

st.title("🏆 2026 FIFA World Cup Predictor")

team1 = st.selectbox("Select Team 1", team_options, index=team_options.index("France"))
team2 = st.selectbox("Select Team 2", team_options, index=team_options.index("Argentina"))


def Rating_Difference(team1, team2):
    idx1 = df[df['team'] == team1].index[0]
    idx2 = df[df['team'] == team2].index[0]

    diff = df.loc[idx1, 'rating'] - df.loc[idx2, 'rating']
    return diff

def value_difference(team1,team2):
    idx1 = df[df['team'] == team1].index[0]
    idx2 = df[df['team'] == team2].index[0]
    diff = df.loc[idx1, 'avg_trans_value'] - df.loc[idx2, 'avg_trans_value']
    return diff

def form_diff(team1,team2):
    idx1 = df[df['team'] == team1].index[0]
    idx2 = df[df['team'] == team2].index[0]
    diff = df.loc[idx1, 'W%'] - df.loc[idx2, 'W%']
    return diff


def age_diff(team1, team2):
    idx1 = df[df['team'] == team1].index[0]
    idx2 = df[df['team'] == team2].index[0]

    diff = df.loc[idx1, 'avg_squad_age'] - df.loc[idx2, 'avg_squad_age']
    return diff


def defense_stability_diff(team1, team2):
    idx1 = df[df['team'] == team1].index[0]
    idx2 = df[df['team'] == team2].index[0]


    avg_ga_t1 = df.loc[idx1, 'GA'] / 5
    avg_ga_t2 = df.loc[idx2, 'GA'] / 5


    return avg_ga_t1 - avg_ga_t2

if st.button("Simulate Match ⚽"):

    match_features = pd.DataFrame([{
        'rating_diff': Rating_Difference(team1, team2),
        'value_diff': value_difference(team1, team2),
        'form_diff': form_diff(team1, team2),
        'age_diff': age_diff(team1, team2),
        'def_stability_diff': defense_stability_diff(team1, team2)
    }])

    base_gd = model.predict(match_features)[0]


    t1_wins, draws, t2_wins = 0, 0, 0
    simulations = 1000

    for _ in range(simulations):

        simulated_gd = np.random.normal(loc=base_gd, scale=1.22)
        if simulated_gd > 0.25:
            t1_wins += 1
        elif simulated_gd < -0.25:
            t2_wins += 1
        else:
            draws += 1


    t1_prob = (t1_wins / simulations) * 100
    draw_prob = (draws / simulations) * 100
    t2_prob = (t2_wins / simulations) * 100

    avg_total_goals = 2.5
    t1_expected = max(0.2, (avg_total_goals + base_gd) / 2)
    t2_expected = max(0.2, (avg_total_goals - base_gd) / 2)

    t1_score = np.random.poisson(lam=t1_expected)
    t2_score = np.random.poisson(lam=t2_expected)


    st.write("---")
    st.subheader("📊 Match Forecast Probabilities")


    st.write(f"**{team1} Win:** {t1_prob:.1f}%")
    st.progress(t1_wins / simulations)


    st.write(f"**Draw:** {draw_prob:.1f}%")
    st.progress(draws / simulations)


    st.write(f"**{team2} Win:** {t2_prob:.1f}%")
    st.progress(t2_wins / simulations)


    st.write("---")
    st.subheader("🎰 Simulated Scoreline")
    st.success(f"🏟️ Final Score: **{team1} {t1_score} - {t2_score} {team2}**")