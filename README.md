# Customer Churn Prediction using Logistic Regression

## Objective
Develop a Logistic Regression model to predict customer churn based on demographic information and service usage.

## Dataset Link
[Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Methodology
1. **Data Understanding:** Loaded the dataset, displayed records, and identified numerical, categorical, and target features.
2. **Data Preprocessing:** Handled missing values (e.g., in `TotalCharges`), encoded categorical variables using one-hot encoding, and split the data into 80% training and 20% testing sets.
3. **Model Development:** Trained a Logistic Regression model using the training data with `Churn` as the target variable.
4. **Model Evaluation:** Evaluated the model using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

## Results
The Logistic Regression model achieved competitive accuracy. The confusion matrix and classification report detail the exact precision, recall, and F1-score for both classes (Churn: Yes and Churn: No).

## Conclusion
- **Key findings:** The model successfully predicts customer churn. Customers with month-to-month contracts, absence of online security, and fiber optic internet services showed higher tendencies to churn.
- **Factors influencing customer churn:** Contract type, tenure, internet service type, and monthly charges were significant indicators of churn.
- **Limitation of Logistic Regression:** Logistic Regression assumes a linear relationship between the log-odds of the dependent variable and the independent variables. It might struggle to capture complex, non-linear relationships among features compared to tree-based models like Random Forest or Gradient Boosting.
