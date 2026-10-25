
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Electrical Equipment Failure Early Warning",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "failure_model.joblib"

# ---------------- HEADER ----------------
st.title("⚡ Electrical Equipment Failure Early Warning")

st.write(
    "An ML-based demonstration system that estimates "
    "equipment failure risk using machine sensor readings."
)

st.info(
    "Dataset: AI4I 2020 Predictive Maintenance Dataset. "
    "This is synthetic demonstration data, not live industrial data."
)

st.divider()

# ---------------- LOAD TRAINED MODEL ----------------
if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. "
        "Run train_evaluate.py first."
    )
    st.stop()

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("About the Project")
st.sidebar.write(
    "This project demonstrates machine failure risk "
    "estimation using a trained Random Forest model."
)

st.sidebar.write("Dataset: AI4I 2020")
st.sidebar.write("Model: Random Forest Classifier")

st.sidebar.warning(
    "Demonstration only. Not validated for real industrial safety decisions."
)

# ---------------- SENSOR INPUTS ----------------
st.header("1. Equipment Sensor Readings")

left, right = st.columns(2)

with left:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"],
        help="Select the machine quality type."
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=298.1,
        step=0.1
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=308.6,
        step=0.1
    )

with right:
    speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=0,
        max_value=10000,
        value=1500,
        step=10
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=500.0,
        value=40.0,
        step=0.5
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=1000,
        value=100,
        step=1
    )

# ---------------- PREDICTION ----------------
st.divider()
st.header("2. Failure Risk Prediction")

if st.button("Predict Failure Risk", type="primary"):

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Match the feature columns used during training
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )

    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)
        failure_index = classes.index(1)
        failure_probability = float(
            probabilities[failure_index]
        )

        st.subheader("Prediction Result")

        metric_col, result_col = st.columns(2)

        with metric_col:
            st.metric(
                "Estimated Failure Probability",
                f"{failure_probability:.1%}"
            )

        with result_col:
            if prediction == 1:
                st.error("⚠️ Model predicts possible failure.")
            else:
                st.success("✅ Model predicts normal operation.")

        st.progress(failure_probability)

        st.caption(
            "This probability is a model estimate, not a guaranteed "
            "failure probability or a verified safety assessment."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()

# ---------------- MODEL EVALUATION ----------------
st.header("3. Model Evaluation")

metrics_path = RESULTS_DIR / "evaluation_metrics.csv"

if metrics_path.exists():
    try:
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True)
    except Exception as error:
        st.warning(f"Could not read evaluation metrics: {error}")
else:
    st.info(
        "Evaluation metrics file has not been created yet. "
        "See the graphs below and the training script output."
    )

# ---------------- EVALUATION GRAPHS ----------------
st.header("4. Evaluation Graphs")

graph_files = [
    ("confusion_matrix.png", "Confusion Matrix"),
    ("feature_importance.png", "Feature Importance"),
    ("failed_cases_graph.png", "Failed Cases Analysis")
]

for filename, title in graph_files:
    graph_path = RESULTS_DIR / filename

    if graph_path.exists():
        st.subheader(title)
        st.image(
            str(graph_path),
            caption=title,
            use_container_width=True
        )
    else:
        st.caption(f"{title} graph is not available yet.")

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Academic ML Project | Electrical Equipment Failure Early Warning"
)

st.caption(
    "Limitations: Uses synthetic dataset examples and offline predictions. "
    "Not intended for real-time monitoring or industrial safety decisions."
)
