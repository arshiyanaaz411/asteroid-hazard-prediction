import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="☄️ Asteroid Hazard Predictor", page_icon="☄️", layout="wide")

st.title("☄️ Asteroid Hazard Predictor")
st.markdown("Asteroid details enter karo aur dekho ki object potentially hazardous hai ya nahi.")

tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Input Summary", "ℹ️ Feature Help"])

with tab1:
    st.sidebar.header("🔧 Input Features")

    abs_mag = st.sidebar.slider("Absolute Magnitude", 10.0, 35.0, 20.0)
    dia_min = st.sidebar.slider("Est Diameter KM (min)", 0.0, 5.0, 0.1)
    velocity = st.sidebar.slider("Relative Velocity (km/s)", 0.0, 50.0, 10.0)
    miss_dist = st.sidebar.slider("Miss Distance (Astronomical)", 0.0, 0.5, 0.2)
    orbit_unc = st.sidebar.slider("Orbit Uncertainty", 0, 9, 3)
    min_orbit = st.sidebar.slider("Minimum Orbit Intersection", 0.0, 0.5, 0.05)
    jt_inv = st.sidebar.slider("Jupiter Tisserand Invariant", 2.0, 10.0, 5.0)
    ecc = st.sidebar.slider("Eccentricity", 0.0, 1.0, 0.3)
    semi_major = st.sidebar.slider("Semi Major Axis", 0.5, 5.0, 1.5)
    incl = st.sidebar.slider("Inclination", 0.0, 90.0, 10.0)
    asc_node = st.sidebar.slider("Asc Node Longitude", 0.0, 360.0, 100.0)
    orb_period = st.sidebar.slider("Orbital Period", 100.0, 2000.0, 500.0)
    peri_dist = st.sidebar.slider("Perihelion Distance", 0.0, 2.0, 0.9)
    peri_arg = st.sidebar.slider("Perihelion Arg", 0.0, 360.0, 150.0)
    mean_anom = st.sidebar.slider("Mean Anomaly", 0.0, 360.0, 180.0)

    features = np.array([[abs_mag, dia_min, velocity, miss_dist, orbit_unc,
                          min_orbit, jt_inv, ecc, semi_major,
                          incl, asc_node, orb_period, peri_dist, peri_arg, mean_anom]])

    col1, col2, col3 = st.columns(3)
    col1.metric("Velocity", f"{velocity:.2f} km/s")
    col2.metric("Miss Distance", f"{miss_dist:.3f} AU")
    col3.metric("Diameter Min", f"{dia_min:.2f} km")

    if st.button("🔍 Predict!"):
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ POTENTIALLY HAZARDOUS ASTEROID!")
            st.metric("Hazard Probability", f"{prob[1]*100:.1f}%")
        else:
            st.success("✅ NOT Hazardous")
            st.metric("Safe Probability", f"{prob[0]*100:.1f}%")

with tab2:
    st.subheader("📊 Current Input Summary")

    input_df = pd.DataFrame({
        "Feature": [
            "Absolute Magnitude", "Est Diameter KM (min)", "Relative Velocity (km/s)",
            "Miss Distance (Astronomical)", "Orbit Uncertainty", "Minimum Orbit Intersection",
            "Jupiter Tisserand Invariant", "Eccentricity", "Semi Major Axis", "Inclination",
            "Asc Node Longitude", "Orbital Period", "Perihelion Distance",
            "Perihelion Arg", "Mean Anomaly"
        ],
        "Value": [
            abs_mag, dia_min, velocity, miss_dist, orbit_unc, min_orbit, jt_inv,
            ecc, semi_major, incl, asc_node, orb_period, peri_dist, peri_arg, mean_anom
        ]
    })

    st.dataframe(input_df, use_container_width=True)

    fig = px.bar(
        input_df,
        x="Feature",
        y="Value",
        title="Asteroid Input Features Overview"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("ℹ️ Feature Help")
    st.markdown("""
    - **Absolute Magnitude**: Asteroid ki brightness.
    - **Est Diameter KM (min)**: Asteroid ka minimum estimated size.
    - **Relative Velocity**: Asteroid kitni speed se move kar raha hai.
    - **Miss Distance**: Earth ke kitne paas se niklega.
    - **Orbit Uncertainty**: Orbit calculation kitni uncertain hai.
    - **Minimum Orbit Intersection**: Earth orbit ke kitne close aata hai.
    - **Eccentricity**: Orbit kitna circular ya stretched hai.
    - **Inclination**: Orbit ka tilt.
    """)
