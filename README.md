# MachineGuard AI

### AI-Based Predictive Maintenance and Early Failure Warning System Using Deep Learning

## Project Overview

MachineGuard AI is a machine failure prediction system that analyzes industrial machine sensor data to identify potential failure risks and support early maintenance decisions.

The application provides an interactive dashboard to explore machine information, sensor readings, failure history, maintenance records, and predicted warning risks.

## Objectives

* Analyze industrial machine telemetry data.
* Identify machines that may require early maintenance.
* Predict potential machine failure risks using deep learning.
* Visualize sensor trends and historical failure information.
* Support preventive maintenance planning and reduce unexpected downtime.

## Dataset

This project uses the Predictive Maintenance (PdM) dataset, containing:

* PdM_telemetry.csv — Machine sensor readings over time.
* PdM_failures.csv — Historical machine failure records.
* PdM_machines.csv — Machine information.
* PdM_maint.csv — Maintenance records.
* PdM_errors.csv — Machine error records.

The dataset is used for educational and predictive maintenance analysis. It should not be interpreted as verified live data from a specific factory.

## Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* Matplotlib and Plotly (where applicable)

## Key Features

* Interactive machine monitoring dashboard
* Machine identification and information
* Failure risk prediction
* Sensor analytics and visualization
* Failure and maintenance history
* Early warning support for maintenance decisions

## Project Structure

MachineGuard-AI/
├── app.py
├── PdM_telemetry.csv
├── PdM_failures.csv
├── PdM_machines.csv
├── PdM_maint.csv
├── PdM_errors.csv
├── machine_failure_model_final.keras
├── machine_failure_scaler_final.pkl
├── machine_failure_features_final.pkl
├── machine_banner.png
└── requirements.txt

## Disclaimer

MachineGuard AI is an educational project and decision-support prototype. Predictions are estimates, not guarantees of machine failure. Maintenance decisions should be validated by qualified personnel and actual machine inspections.

## Project Purpose

This project demonstrates the application of data analysis, deep learning, and interactive dashboards to predictive maintenance.

## Conclusion

MachineGuard AI demonstrates how deep learning can analyze machine sensor data to identify potential failure risks and support early maintenance decisions. By combining sensor readings, machine information, failure history, and maintenance records, the system helps identify machines that may require attention. The interactive Streamlit dashboard supports machine monitoring, risk prediction, and maintenance planning, helping reduce unexpected downtime and improve machine reliability. However, predictions should be validated before making actual maintenance decisions.

⸻

MachineGuard AI | Predictive Maintenance Using Deep Learning
