# Customer Churn Prediction Project

## Project Overview

Customer churn—the phenomenon where customers discontinue their subscription to a service—poses a significant challenge for subscription-based businesses. Accurately predicting which customers are likely to leave enables companies to proactively engage at-risk customers, optimize retention strategies, and ultimately improve profitability.

This project aims to develop a predictive model using machine learning techniques such as Random Forest and LightGBM, combined with thoughtful feature engineering, to identify customers at high risk of churn. The insights derived will inform actionable business recommendations and guide retention efforts.

## Business Understanding

For subscription-based businesses, retaining existing customers is often more cost-effective than acquiring new ones. High churn rates can signal underlying issues with customer satisfaction, product fit, or competitive pressures, directly impacting revenue and growth.

By understanding the drivers of churn and identifying at-risk customers, the business can:
- Target retention campaigns more effectively (e.g., personalized offers, improved customer support)
- Allocate resources efficiently to maximize return on investment in retention
- Reduce lost revenue and improve customer lifetime value

### Key Business Questions
- Which customers are most likely to leave the service in the near future?
- What are the main factors contributing to customer churn?
- How can the business intervene to reduce churn, and what is the expected financial impact of these interventions?

## Dataset

The project uses the **Telco Customer Churn** dataset containing historical customer data including:
- **Demographics**: gender, senior citizen status, partner status, dependents
- **Service Information**: phone service, internet service, online security, tech support
- **Account Details**: tenure, contract type, paperless billing, payment method
- **Charges**: monthly charges, total charges
- **Target Variable**: churn status (Yes/No)

**Dataset Size**: 7,043 customers with 21 features

## Methodology

### 1. Data Preprocessing

#### Data Cleaning
- Converted 'TotalCharges' column from object to numeric type
- Handled missing values (11 rows dropped after conversion)
- Identified categorical and numerical columns for encoding and scaling

#### Feature Engineering
- Applied one-hot encoding to categorical features (excluding customerID)
- Used StandardScaler for numerical features (tenure)
- Prepared data for machine learning algorithms

### 2. Feature Selection

#### Correlation Analysis
- Calculated correlation matrix between features and target variable
- Selected features with absolute correlation > 0.1 with 'Churn_Yes'
- Identified 21 most relevant features for prediction

#### Selected Features
Key features include:
- Internet Service (Fiber optic)
- Payment Method (Electronic check, Credit card automatic)
- Monthly Charges
- Paperless Billing
- Senior Citizen status
- Partner and Dependents status
- Tech Support and Online Security
- Contract type (One year, Two year)
- Total Charges
- Tenure

### 3. Model Selection

#### Evaluated Models
1. **Logistic Regression**
   - Pros: Simple, interpretable, computationally efficient
   - Cons: Assumes linearity, may miss complex relationships

2. **Decision Trees**
   - Pros: Easy to interpret, handles mixed data types
   - Cons: Prone to overfitting, unstable

3. **Random Forest**
   - Pros: Reduces overfitting, good performance, feature importance
   - Cons: Less interpretable than single trees

4. **Gradient Boosting (LightGBM)**
   - Pros: State-of-the-art performance, captures complex relationships
   - Cons: Can overfit, less interpretable

#### Chosen Models
- **Random Forest**: Good balance of performance and reduced overfitting
- **LightGBM**: Excellent performance and relatively fast training

### 4. Model Training

- Split data: 80% training, 20% testing
- Applied Random Forest and LightGBM classifiers
- Used random_state=42 for reproducibility

### 5. Model Evaluation

#### Performance Metrics
Both models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

#### Results Comparison

| Metric | Random Forest | LightGBM |
|--------|---------------|----------|
| Accuracy | 0.7846 | 0.7910 |
| Precision | 0.6246 | 0.6351 |
| Recall | 0.4759 | 0.5027 |
| F1 Score | 0.5402 | 0.5612 |
| ROC AUC | 0.8155 | 0.8253 |

**Winner**: LightGBM performed slightly better across all metrics

### 6. Feature Importance Analysis

#### Top 10 Most Important Features (LightGBM)
1. **MonthlyCharges** (809) - Highest importance
2. **TotalCharges** (658) - Second highest
3. **tenure** (467) - Customer loyalty indicator
4. **gender_Male** (110)
5. **PaymentMethod_Electronic check** (79)
6. **PaperlessBilling_Yes** (78)
7. **OnlineSecurity_Yes** (65)
8. **OnlineBackup_Yes** (65)
9. **Dependents_Yes** (59)
10. **SeniorCitizen** (59)

## Key Findings

### Data Analysis Insights
- The 'TotalCharges' column required type conversion from object to numeric
- No missing values in the original dataset
- Feature selection identified 21 highly correlated features with churn
- LightGBM outperformed Random Forest, suggesting complex non-linear relationships

### Business Insights
- **Monthly Charges** and **Total Charges** are the strongest predictors of churn
- **Tenure** (customer loyalty) significantly impacts churn probability
- **Payment methods** and **billing preferences** influence customer retention
- **Service features** like online security and backup affect churn rates

## Recommendations

### Immediate Actions
1. **Monitor High-Risk Customers**: Focus on customers with high monthly charges and low tenure
2. **Payment Method Optimization**: Investigate why electronic check users have higher churn rates
3. **Service Bundle Analysis**: Review pricing strategies for fiber optic and premium services

### Strategic Initiatives
1. **Retention Programs**: Develop targeted campaigns for customers with specific risk factors
2. **Service Enhancement**: Improve online security and backup services to reduce churn
3. **Pricing Strategy**: Review pricing models to balance revenue and retention

### Model Improvements
1. **Hyperparameter Tuning**: Further optimize LightGBM parameters
2. **Feature Engineering**: Create interaction features between key variables
3. **Ensemble Methods**: Combine multiple models for improved performance

## Technical Requirements

### Dependencies
```python
pandas
numpy
scikit-learn
lightgbm
seaborn
matplotlib
```

### Data Requirements
- Clean, structured customer data
- No missing values in critical features
- Proper data types for numerical and categorical variables

## Future Work

1. **Model Deployment**: Implement the trained model in production
2. **Real-time Scoring**: Develop API for real-time churn prediction
3. **A/B Testing**: Test retention strategies based on model predictions
4. **Model Monitoring**: Track model performance over time
5. **Feature Updates**: Incorporate new customer behavior data

## Conclusion

This customer churn prediction project successfully demonstrates the application of machine learning to business problems. The LightGBM model achieved an ROC AUC of 0.8253, providing reliable predictions for customer churn risk.

The analysis reveals that pricing (monthly and total charges) and customer loyalty (tenure) are the primary drivers of churn. These insights can guide targeted retention strategies and improve customer lifetime value.

By implementing the recommendations and continuously monitoring model performance, businesses can proactively address customer churn and improve overall profitability.