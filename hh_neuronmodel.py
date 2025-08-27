import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------- HEADER --------------------
st.title("Whole-Cell HH Simulator with Kinetics & IV Curves")

st.markdown("""
Simulate action potentials using realistic patch-clamp units.  
Adjust gating kinetics and see both the AP, phase plot, and IV curves.
""")

# -------------------- COLLAPSIBLE ION CONCENTRATIONS --------------------
with st.expander("Ion Concentrations & GHK Parameters"):
    st.write("Used to initialize resting potential")

    Na_in  = st.slider("Intracellular [Na+] (mM)", 0, 50, 15)
    Na_out = st.slider("Extracellular [Na+] (mM)", 0, 200, 145)
    K_in   = st.slider("Intracellular [K+] (mM)", 0, 200, 150)
    K_out  = st.slider("Extracellular [K+] (mM)", 0, 20, 5)
    Cl_in  = st.slider("Intracellular [Cl-] (mM)", 0, 50, 10)
    Cl_out = st.slider("Extracellular [Cl-] (mM)", 0, 150, 110)

    P_Na = st.slider("Na+ permeability", 0.0, 1.0, 0.05)
    P_K  = st.slider("K+ permeability", 0.0, 1.0, 1.0)
    P_Cl = st.slider("Cl- permeability", 0.0, 1.0, 0.45)

# -------------------- RESTING POTENTIAL --------------------
R = 8.314
T = 310
F = 96485
V_rest = (R * T / F) * np.log(
    (P_K * K_out + P_Na * Na_out + P_Cl * Cl_in) /
    (P_K * K_in  + P_Na * Na_in  + P_Cl * Cl_out)
) * 1000  # in mV

# -------------------- HH PARAMETERS --------------------
st.sidebar.header("HH Channel Parameters (Whole-cell)")

C_m = st.sidebar.slider("Membrane capacitance C_m (pF)", 5.0, 100.0, 50.0)

g_Na = st.sidebar.slider("Max Na+ conductance g_Na (nS)", 1.0, 500.0, 310.0)
g_K  = st.sidebar.slider("Max K+ conductance g_K (nS)", 10.0, 500.0, 200.0)
g_L  = st.sidebar.slider("Leak conductance g_L (nS)", 1.0, 20.0, 10.0)

E_Na = st.sidebar.slider("Na+ reversal potential E_Na (mV)", -30.0, 280.0, 120.0)
E_K  = st.sidebar.slider("K+ reversal potential E_K (mV)", -100.0, 50.0, -40.0)
E_L  = st.sidebar.slider("Leak reversal potential E_L (mV)", -70.0, -50.0, -40.4)

# -------------------- STIMULUS --------------------
st.sidebar.header("Current Injection (pA)")

I_amp   = st.sidebar.slider("Amplitude (pA)", 0.0, 1000.0, 20.0)
I_start = st.sidebar.slider("Start time (ms)", 0.0, 500.0, 50.0)
I_dur   = st.sidebar.slider("Duration (ms)", 0.1, 500.0, 2.0)

# -------------------- KINETICS SLIDERS --------------------
st.sidebar.header("Channel Kinetics Multipliers")

m_alpha_mult = st.sidebar.slider("m activation rate multiplier", 0.1, 20.0, 10.0)
h_alpha_mult = st.sidebar.slider("h inactivation rate multiplier", 0.1, 10.0, 1.0)
n_alpha_mult = st.sidebar.slider("n activation rate multiplier", 0.01, 5.0, 1.5)

# -------------------- TIME --------------------
dt    = 0.01  # ms
t_max = 500.0
time_hh = np.arange(0.0, t_max, dt)

I_ext = np.zeros_like(time_hh)
I_ext[(time_hh >= I_start) & (time_hh < I_start + I_dur)] = I_amp

# -------------------- INITIALIZATION --------------------
V = np.zeros_like(time_hh)
V[0] = V_rest

m = np.zeros_like(time_hh)
h = np.zeros_like(time_hh)
n = np.zeros_like(time_hh)

V0 = V[0]

# Initial gating values
alpha_m = 0.1 * (25 - V0) / (np.exp((25 - V0) / 10) - 1)
beta_m  = 4 * np.exp(-V0 / 18)
m[0] = alpha_m / (alpha_m + beta_m)
h[0] = 0.6

alpha_n = 0.01 * (10 - V0) / (np.exp((10 - V0) / 10) - 1)
beta_n  = 0.125 * np.exp(-V0 / 80)
n[0] = alpha_n / (alpha_n + beta_n)

# -------------------- SIMULATION --------------------
for i in range(1, len(time_hh)):
    V_prev = V[i - 1]

    # Gating variables
    alpha_m = 0.1 * (25 - V_prev) / (np.exp((25 - V_prev) / 10) - 1) * m_alpha_mult
    beta_m  = 4 * np.exp(-V_prev / 18)
    m[i] = m[i - 1] + dt * (alpha_m * (1 - m[i - 1]) - beta_m * m[i - 1])

    alpha_h = 0.07 * np.exp(-V_prev / 20) * h_alpha_mult
    beta_h  = 1 / (np.exp((30 - V_prev) / 10) + 1)
    h[i] = h[i - 1] + dt * (alpha_h * (1 - h[i - 1]) - beta_h * h[i - 1])

    alpha_n = 0.01 * (10 - V_prev) / (np.exp((10 - V_prev) / 10) - 1) * n_alpha_mult
    beta_n  = 0.125 * np.exp(-V_prev / 80)
    n[i] = n[i - 1] + dt * (alpha_n * (1 - n[i - 1]) - beta_n * n[i - 1])

    # Currents in pA
    I_Na = g_Na * (m[i]**3) * h[i] * (V_prev - E_Na)
    I_K  = g_K  * (n[i]**4) * (V_prev - E_K)
    I_L  = g_L  * (V_prev - E_L)

    # Voltage update
    V[i] = V_prev + dt * (I_ext[i] - I_Na - I_K - I_L) / C_m

# -------------------- PLOTS --------------------

# 1️⃣ Action potential
x_min_ap, x_max_ap = st.slider("Time axis range (ms)", 0.0, 500.0, (0.0, 500.0), step=1.0)
y_min_ap, y_max_ap = st.slider("Voltage axis range (mV)", -100.0, 100.0, (-80.0, 50.0), step=1.0)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time_hh, V, color='red', label='Membrane Potential')
ax.plot(time_hh, I_ext*0.05 - 70, '--', color='blue', label='Injected Current (scaled)')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("V (mV)")
ax.set_title("Whole-cell HH Action Potential")
ax.set_xlim(x_min_ap, x_max_ap)
ax.set_ylim(y_min_ap, y_max_ap)
ax.grid(True)
ax.legend()
st.pyplot(fig)

# 2️⃣ Phase plot V vs n
st.subheader("Phase Plot: V vs n (Potassium gating)")

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(V, n, color='purple')
ax2.set_xlabel("Membrane Potential V (mV)")
ax2.set_ylabel("n (K+ activation)")
ax2.set_title("Phase Plot")
ax2.grid(True)
st.pyplot(fig2)

# 3️⃣ IV curves
st.subheader("Voltage-Clamp IV Curves (Dynamic with Kinetics)")

V_range = np.linspace(-80, 40, 200)
V_shift = V_range - V_rest

# Steady-state gating
alpha_m = 0.1*(25 - V_shift)/(np.exp((25 - V_shift)/10) - 1) * m_alpha_mult
beta_m  = 4*np.exp(-V_shift/18)
m_inf   = (alpha_m / (alpha_m + beta_m))**3

alpha_h = 0.07*np.exp(-V_shift/20) * h_alpha_mult
beta_h  = 1/(np.exp((30 - V_shift)/10) + 1)
h_inf   = alpha_h / (alpha_h + beta_h)

alpha_n = 0.01*(10 - V_shift)/(np.exp((10 - V_shift)/10) - 1) * n_alpha_mult
beta_n  = 0.125*np.exp(-V_shift/80)
n_inf   = (alpha_n / (alpha_n + beta_n))**4

I_Na_IV = g_Na * m_inf * h_inf * (V_range - E_Na)
I_K_IV  = g_K  * n_inf * (V_range - E_K)

fig3, ax3 = plt.subplots(figsize=(6,4))
ax3.plot(V_range, I_Na_IV, label='I_Na')
ax3.plot(V_range, I_K_IV, label='I_K')
ax3.set_xlabel("Voltage (mV)")
ax3.set_ylabel("Current (pA)")
ax3.set_title("Voltage-Clamp IV Curves")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)
