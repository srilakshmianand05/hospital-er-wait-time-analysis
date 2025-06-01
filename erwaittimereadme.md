🏥 ER Wait Time Analysis & Prediction with Machine Learning

📊 Project Overview
Emergency Rooms (ERs) are critical healthcare access points, but excessive wait times can significantly impact patient outcomes and satisfaction. This project presents a comprehensive analysis of ER wait times, combining data preprocessing, exploratory data analysis (EDA), outlier treatment, and Random Forest Regression modeling to predict total wait time for patients. It also features interactive Power BI dashboards providing actionable insights for hospital administrators and healthcare decision-makers.

📁 Dataset Summary
Source: Simulated ER wait time dataset
Shape: 5000 rows × 19 columns

Features Included:

Patient demographics and identifiers (dropped during modeling)

Operational features: Facility Size, Specialist Availability, Time to Registration, Time to Triage, Time to Medical Professional

Target variable: Total Wait Time

Outcome indicators and hospital-level data

🔧 Key Steps in the Analysis

1. Data Cleaning and Preprocessing

Null value check and removal

Duplicate record elimination

Outlier handling using:

IQR Method

Z-Score Method

Winsorization for extreme values

Label Encoding for categorical (object) features

Dropped identifiers: Visit ID, Patient ID, Hospital ID, Hospital Name, Visit Date

2. Feature Engineering & EDA

Correlation Matrix to analyze relationships between:

Specialist Availability

Facility Size

Time to Registration

Time to Triage

Time to Medical Professional

Total Wait Time

Insights visualized using Power BI (refer to dashboards below)

🤖 Machine Learning Model: Predicting ER Wait Time

🎯 Problem Statement
The goal is to predict the total ER wait time for a patient using operational, clinical, and hospital-specific variables. Accurate predictions can help hospitals manage resource allocation, reduce patient frustration, and improve overall satisfaction.

📌 Target Variable
Total Wait Time (in minutes) – the sum of time taken for registration, triage, and medical assessment.

🧹 Data Preparation for Modeling
python
Copy
Edit
# Feature-target split
X = df.drop('Total Wait Time', axis=1)
y = df['Total Wait Time']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Independent Variables: Operational and facility metrics (e.g., Time to Registration, Time to Triage, Facility Size, Specialist Availability, Urgency Level, etc.)

Excluded Features: Identifiers (Visit ID, Patient ID, Hospital ID) and date fields to avoid data leakage.

🤖 Model Selection: Random Forest Regressor
Random Forest was chosen for its:

High accuracy in regression tasks

Robustness to outliers

Ability to model complex nonlinear relationships

Automatic feature importance ranking

python
Copy
Edit
from sklearn.ensemble import RandomForestRegressor

# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
📊 Model Evaluation
python
Copy
Edit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
Metric	Score
R² Score	0.997
Mean Absolute Error	1.998 minutes
Root Mean Squared Error	~2.15 minutes

✅ These scores reflect excellent predictive accuracy with very low error in real-world terms (average deviation < 2 minutes).

🔍 Feature Importance
python
Copy
Edit
import matplotlib.pyplot as plt
import pandas as pd

# Extract and plot
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6), title='Feature Importance')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()
Top Features Influencing Wait Time:

Time to Medical Professional

Time to Triage

Time to Registration

Urgency Level

Specialist Availability

These insights not only help in predictive modeling, but also guide operational interventions to reduce delays.

📉 Predicted vs Actual Comparison
python
Copy
Edit
import seaborn as sns

# Plotting predicted vs actual wait time
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Wait Time")
plt.ylabel("Predicted Wait Time")
plt.title("Actual vs. Predicted ER Wait Time")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.tight_layout()
plt.show()
A nearly perfect alignment along the diagonal indicates a well-calibrated model with high generalizability.

✅ Summary
Model Used: RandomForestRegressor from sklearn

Accuracy: 99.7%

Key Factors: Time to medical services, urgency, and facility operations

Deployment Ready: Model can be exported as a .pkl file for real-time applications in a healthcare system
📊 Dashboard Insights (Power BI)
🔹 Dashboard 1: Hospital Wait Time Analysis & Patient Care Insights
Satisfaction & Outcome by Hospital

Monthly Wait Time Trends by Region

Wait Time by Day of Week

Patient Outcome by Urgency Level

Specialist Availability & Nurse Ratio

Seasonal Wait Time Distribution

🔹 Dashboard 2: Regression Predictions & Operational Trends
Predicted vs. Actual Wait Times

Wait Time by Registration Time & Urgency

Time Distribution by ER Stages

Average Wait Time by Satisfaction Level

Urgency Level vs. Predicted/Actual Time

These dashboards deliver high-impact insights into operational bottlenecks and how they influence satisfaction and clinical outcomes.

🧠 Key Learnings
Applied advanced outlier detection techniques to real-world hospital data

Built a high-accuracy predictive model for patient wait time estimation

Gained experience with feature importance interpretation

Created professional-grade dashboards for operational insight communication

💼 Potential Applications
Hospital Resource Management

Patient Satisfaction Improvement

Predictive Triage Optimization

Operational Workflow Planning

📌 Tools & Technologies
Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)

Power BI for interactive data visualization

Jupyter Notebook for exploratory data analysis and modeling

🚀 Future Enhancements
Integrate live hospital data pipelines for real-time monitoring

Deploy model via Flask API for real-time wait time prediction

Introduce classification models to predict satisfaction scores

Include geospatial analysis for regional performance comparison

📬 Contact
For feedback, collaborations, or questions:

Srilakshmi
Aspiring Data Analyst | Python | SQL | Power BI | Excel | Machine Learning
📧 www.asrilakshmi8897@gmail.com
🔗 www.linkedin.com/in/asrilakshmi0509 

