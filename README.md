# Telecom Customer Churn Prediction

## Project Overview
This project predicts customer churn for a telecom company using machine learning and visualizes actionable insights through Power BI dashboards. The goal is to identify customers likely to leave the service so that proactive retention strategies can be implemented.

## Tech Stack / Tools
- Programming & ML: Python, Pandas, NumPy, Scikit-learn, XGBoost
- Data Visualization: Power BI
- Model Persistence: Pickle

## Dataset
- Source: Kaggle – Telco Customer Churn
- Key features include:
  - Customer demographics (gender, SeniorCitizen, Partner)
  - Service information (Contract, InternetService, PaymentMethod)
  - Billing information (MonthlyCharges, TotalCharges, tenure)
- Target variable: Churn (Yes = 1, No = 0)

## Project Workflow
1. **Data Cleaning:**
   - Converted TotalCharges to numeric
   - Filled missing values with median
   - Dropped customerID as it is not predictive

2. **Feature Preparation:**
   - Separated numeric and categorical features
   - Applied OneHotEncoding for categorical variables

3. **Train-Test Split:**
   - 80% training, 20% testing
   - Stratified split to preserve churn ratio

4. **Modeling:**
   - Trained an XGBoost classifier
   - Handled class imbalance using scale_pos_weight

5. **Evaluation:**
   - Metrics used: Accuracy, Precision, Recall, F1-score, Confusion Matrix
   - Achieved ~75% accuracy on test data

6. **Power BI Dashboard:**
   - Interactive visualizations for:
     - Churn by Contract, Gender, Tenure, and Monthly Charges
     - Filters / Slicers for demographic and service features
   - Provides actionable business insights for customer retention

7. **Model Export:**
   - Saved trained model using pickle
   - Cleaned dataset exported as CSV for visualization

## Key Insights
- Month-to-month contract customers have the highest churn rate
- Customers with longer tenure are less likely to churn
- Higher monthly charges sometimes correlate with higher churn probability
- Dashboard allows filtering by gender, contract type, payment method to identify at-risk segments

## Files in Repository
| File | Description |
|------|-------------|
| churn_model.py | Python script for cleaning data, training XGBoost model, and saving model |
| cleaned_telecom_data.csv | Cleaned dataset used for ML and Power BI dashboard |
| churn_xgb_model.pkl | Saved trained XGBoost model |
| Telecom_Churn_Dashboard.pbix | Power BI dashboard file for interactive visualization |
| README.md | Project overview and instructions |

## How to Run
1. Clone the repository:
git clone <your-repo-url>

2. Install required Python packages:
pip install pandas numpy scikit-learn xgboost

3. Run the Python script:
python churn_model.py

4. Open Power BI Desktop and load `cleaned_telecom_data.csv` to explore the dashboard.

## Next Steps / Improvements
- Improve model accuracy with CatBoost or LightGBM
- Feature engineering: create tenure bins, monthly charges bins, interaction features
- Visualize predicted churn vs actual churn in Power BI
- Deploy model as a web app using Flask / Streamlit

## Outcome
This project demonstrates the ability to:
- Build an end-to-end ML pipeline
- Handle imbalanced datasets and categorical features
- Generate interactive dashboards to communicate insights
- Apply Python + XGBoost + Power BI in a real-world business scenario
