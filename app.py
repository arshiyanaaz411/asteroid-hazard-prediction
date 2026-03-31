import streamlit as st
import pickle
import numpy as np

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="☄️ Asteroid Hazard Predictor", page_icon="☄️")
st.title("☄️ Asteroid Hazard Predictor")
st.markdown("Enter asteroid details to predict if it's **Potentially Hazardous**!")

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
                      min_orbit, jt_inv, 2458000.5, ecc, semi_major,
                      incl, asc_node, orb_period, peri_dist, peri_arg, mean_anom]])

if st.button("🔍 Predict!"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    if prediction == 1:
        st.error("⚠️ POTENTIALLY HAZARDOUS ASTEROID!")
        st.metric("Hazard Probability", f"{prob[1]*100:.1f}%")
    else:
        st.success("✅ NOT Hazardous")
        st.metric("Safe Probability", f"{prob[0]*100:.1f}%")
