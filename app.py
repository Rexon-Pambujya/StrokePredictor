import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model and pipeline
@st.cache_resource
def load_pipeline(path="lr_stroke_pipeline.pkl"):
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipe = load_pipeline()
model = pipe["model"]
scaler = pipe["scaler"]
ohe = pipe["onehot"]
num_cols = pipe["num_cols"]
cat_cols = pipe["cat_cols"]
threshold = pipe["threshold"]

# --- Sidebar Inputs ---
st.sidebar.header("🧾 Patient Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.slider("Age", 0, 120, 50)
hypertension = st.sidebar.radio("Hypertension", [0, 1])
heart_disease = st.sidebar.radio("Heart Disease", [0, 1])
ever_married = st.sidebar.radio("Ever Married", ["Yes", "No"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence = st.sidebar.radio("Residence Type", ["Urban", "Rural"])
avg_glucose = st.sidebar.slider("Avg Glucose Level", 50.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])

# --- Prediction ---
input_dict = {
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'avg_glucose_level': [avg_glucose],
    'bmi': [bmi],
    'gender': [gender],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence],
    'smoking_status': [smoking_status],
}
df_input = pd.DataFrame(input_dict)

X_num = scaler.transform(df_input[num_cols])
X_cat = ohe.transform(df_input[cat_cols])
X = np.hstack([X_num, X_cat])

proba = model.predict_proba(X)[0]
pred_class = "⚠️ Stroke Risk" if proba[1] >= threshold else "✅ No Stroke"

# --- Output ---
st.title("🧠 Stroke Risk Predictor")
st.markdown("Enter patient info in the sidebar →")

st.subheader("🔎 Prediction Result")
st.markdown(f"**Prediction Threshold:** `{threshold}`")
st.markdown(f"### **Prediction:** {pred_class}")

st.write(f"**Probability of Stroke:** `{proba[1]*100:.2f}%`")
st.write(f"**Probability of No Stroke:** `{proba[0]*100:.2f}%`")

# --- Graph ---
st.subheader("📊 Stroke Probability Comparison")
fig, ax = plt.subplots(figsize=(6, 2.5))
bars = ax.barh(['Stroke', 'No Stroke'], [proba[1], proba[0]], color=['#e74c3c', '#27ae60'])

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
            f'{width * 100:.2f}%', va='center', fontsize=10)

ax.set_xlim(0, 1.2)
ax.set_title("Patient's Stroke Risk", fontsize=12)
ax.set_xlabel("Probability")
ax.set_xticks([])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Stroke', 'No Stroke'])
for spine in ax.spines.values():
    spine.set_visible(False)

st.pyplot(fig)



# Plot distribution with patient marker (enhanced)
st.subheader("📈 Stroke Probability Distribution vs Population")

fig, ax = plt.subplots(figsize=(8, 4))
n, bins, patches = ax.hist(
    proba, bins=50, density=True, alpha=0.7, color="#3498db", edgecolor="white", label="Population Distribution"
)

# Color bars based on value
for patch, center in zip(patches, 0.5 * (bins[1:] + bins[:-1])):
    if center >= proba[1]:
        patch.set_facecolor("#e74c3c")  # Red for higher-risk area
    else:
        patch.set_facecolor("#2ecc71")  # Green for lower-risk area

# Add vertical line for patient probability
ax.axvline(proba[1], color='black', linestyle='--', linewidth=2, label="Patient Risk")

# Annotate patient risk
ax.text(proba[1] + 0.01, ax.get_ylim()[1]*0.9,
        f'Patient\n{proba[1]*100:.1f}%',
        color='black', fontsize=10, weight='bold')

# Labels and legend
ax.set_xlabel("Predicted Stroke Probability")
ax.set_ylabel("Density")
ax.set_title("Patient Risk vs. Population", fontsize=14)
ax.legend()

# Style tweaks
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
fig.tight_layout()

st.pyplot(fig)

# --- Personalized Insights ---
st.subheader("🩺 Personalized Risk Factors")

# Real population averages (compute once from your full dataset)
avg_values = {
    "age":              56.8,
    "avg_glucose_level":150.6,
    "bmi":              29.5,
    "hypertension":     0.3,  # 30% prevalence
    "heart_disease":    0.4   # 40% prevalence
}

warnings = []
if age > avg_values["age"] + 10:
    warnings.append(f"🔸 **Age** is above average ({age} vs {avg_values['age']}).")
if hypertension == 1:
    warnings.append("🔸 **Hypertension** present. Control BP via diet, meds, monitoring.")
if heart_disease == 1:
    warnings.append("🔸 **Heart disease** detected. Regular cardiac care recommended.")
if avg_glucose > avg_values["avg_glucose_level"] + 20:
    warnings.append(f"🔸 **Glucose** high ({avg_glucose} vs {avg_values['avg_glucose_level']}).")
if bmi > avg_values["bmi"] + 3:
    warnings.append(f"🔸 **BMI** elevated ({bmi} vs {avg_values['bmi']}).")
if smoking_status == "smokes":
    warnings.append("🔸 **Smoking** detected. Quitting reduces stroke risk.")

if not warnings:
    st.success("✅ No major modifiable risk factors flagged. Keep it up!")
else:
    st.warning("🚨 The following may increase your stroke risk:")
    for w in warnings:
        st.markdown(w)

# --- Radar Chart ---
st.subheader("📊 Relative Risk Radar Chart")

# Categories for radar
categories = ["Age", "Glucose", "BMI", "Hypertension", "Heart Disease"]
N = len(categories)

# Patient vs population ratios
patient_vals = [
    age               / avg_values["age"],
    avg_glucose       / avg_values["avg_glucose_level"],
    bmi               / avg_values["bmi"],
    (hypertension or 0.01) / avg_values["hypertension"],
    (heart_disease or 0.01) / avg_values["heart_disease"]
]
# Use a small non-zero for 0 to plot on log scale if desired else keep linear

# Baseline (population = 1.0)
pop_vals = [1] * N

# Close the loop
patient_vals += patient_vals[:1]
pop_vals     += pop_vals[:1]
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig2, ax2 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax2.plot(angles, pop_vals, color="#3498db", linewidth=2, label="Population Avg")
ax2.fill(angles, pop_vals,  alpha=0.1, color="#3498db")
ax2.plot(angles, patient_vals, color="#e74c3c", linewidth=2, label="Patient")
ax2.fill(angles, patient_vals,  alpha=0.1, color="#e74c3c")

# Fix axes
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories)
ax2.set_rlabel_position(30)
ax2.set_yticks([0.5, 1, 1.5, 2])
ax2.set_yticklabels(["0.5×","1×","1.5×","2×"])
ax2.set_ylim(0, max(max(patient_vals),1.5))

ax2.set_title("Patient vs. Population (Ratios)", y=1.1)
ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
st.pyplot(fig2)
