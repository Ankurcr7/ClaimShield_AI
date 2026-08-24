# 🛡️ ClaimShield AI — Intelligent Insurance Fraud Detection Platform

> **A machine learning-based insurance claim fraud detection and risk assessment system built by Team HexaRisk.**

ClaimShield AI is designed to help insurance companies identify potentially fraudulent claims, prioritize high-risk cases, estimate potential financial loss, and support investigators with data-driven recommendations.

Instead of simply giving a **Fraud / Not Fraud** prediction, the system produces a complete decision-support output:

* 🔍 Fraud probability
* 📊 Fraud risk score
* 🚨 Fraud prediction
* ⚠️ Risk level
* ✅ Recommended action
* 💰 Potential financial loss
* 📈 Feature importance and explainability
* 📋 High-risk claim prioritization
* 📁 Predictions for new, unlabelled claim data

---

# 📌 Project Overview

Insurance fraud can cause significant financial losses and requires investigators to manually review a large number of claims. Reviewing every claim with the same priority is inefficient.

**ClaimShield AI** uses machine learning to analyze insurance claim information and estimate the probability that a claim may be fraudulent.

For example:

| Claim   | Fraud Probability | Risk Score | Prediction           | Recommended Action |
| ------- | ----------------: | ---------: | -------------------- | ------------------ |
| Claim A |              0.12 |         12 | Genuine              | APPROVE            |
| Claim B |              0.55 |         55 | Depends on threshold | UNDER REVIEW       |
| Claim C |              0.88 |         88 | Fraudulent           | INVESTIGATE        |

This allows investigators to focus their attention on claims that require further investigation.

---

# 🎯 Problem Statement

Insurance companies process a large number of claims. Manual fraud detection has several limitations:

* It is time-consuming.
* Investigators may have limited resources.
* Suspicious claims may not be reviewed quickly.
* High-value fraudulent claims can cause major financial losses.
* Simple rule-based systems may miss complex patterns.

The goal of ClaimShield AI is to build an intelligent system that can:

1. Analyze historical insurance claim data.
2. Detect potentially fraudulent claims.
3. Generate a fraud probability for every claim.
4. Convert probabilities into understandable risk levels.
5. Recommend an appropriate business action.
6. Estimate the potential financial exposure of risky claims.

---

# 🚀 Key Features

## 1. Fraud Detection

The model predicts the probability that an insurance claim belongs to the fraudulent class.

```text
Fraud Probability = 0.87
```

A higher probability indicates that the claim appears more suspicious according to the patterns learned by the model.

---

## 2. Fraud Risk Score

Fraud probability is converted into a more business-friendly score between 0 and 100.

```python
Fraud Risk Score = Fraud Probability × 100
```

### Example

```text
Fraud Probability: 0.72
Fraud Risk Score: 72
```

This makes the model output easier for non-technical users and investigators to understand.

---

## 3. Risk-Based Claim Classification

ClaimShield AI categorizes claims into different risk levels.

| Fraud Probability | Risk Level |
| ----------------- | ---------- |
| Below 0.30        | LOW        |
| 0.30 – 0.69       | MEDIUM     |
| 0.70 – 0.89       | HIGH       |
| 0.90 and above    | CRITICAL   |

### Example

```text
Fraud Probability: 0.15 → LOW
Fraud Probability: 0.48 → MEDIUM
Fraud Probability: 0.82 → HIGH
Fraud Probability: 0.95 → CRITICAL
```

---

## 4. Recommended Action System

The model does not stop at prediction. It also suggests an action.

| Probability    | Recommended Action |
| -------------- | ------------------ |
| Below 0.30     | APPROVE            |
| 0.30 – 0.69    | REVIEW             |
| 0.70 and above | INVESTIGATE        |

### Example

```text
Claim 1 → Probability: 0.18 → APPROVE
Claim 2 → Probability: 0.52 → REVIEW
Claim 3 → Probability: 0.84 → INVESTIGATE
```

This helps transform a machine learning prediction into a practical business workflow.

---

## 5. Potential Financial Loss Estimation

The project estimates the potential financial exposure associated with a claim.

```python
Potential Financial Loss = Fraud Probability × Claim Amount
```

### Example

If:

```text
Fraud Probability = 0.70
Claim Amount = ₹100,000
```

Then:

```text
Potential Financial Loss = 0.70 × 100,000
                         = ₹70,000
```

This helps prioritize claims based not only on fraud probability but also on possible financial impact.

A claim with a moderate fraud probability but a very large claim amount may deserve more attention than a highly suspicious claim with a very small amount.

---

# 🧠 Machine Learning Pipeline

The overall project workflow is:

```text
Insurance Claim Dataset
        │
        ▼
Data Understanding & Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Missing Value Handling
        │
        ▼
Leakage Detection
        │
        ▼
Feature Engineering
        │
        ▼
Preprocessing
        │
        ├── Numeric Features → Median Imputation
        │
        └── Categorical Features → Most Frequent Imputation
                                  + One-Hot Encoding
        │
        ▼
Model Training
        │
        ├── Random Forest
        │
        └── XGBoost
        │
        ▼
Cross-Validation
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Threshold Optimization
        │
        ▼
Tuned XGBoost Final Model
        │
        ▼
Fraud Probability
        │
        ├── Fraud Risk Score
        ├── Fraud Prediction
        ├── Risk Level
        ├── Recommended Action
        └── Potential Financial Loss
```

---

# 📂 Dataset Features

The production model uses the following original features.

## Customer and Policy Information

* `Age`
* `Gender`
* `Region`
* `Policy_Type`

## Financial Information

* `Premium_Amount`
* `Coverage_Amount`
* `Claim_Amount`
* `Credit_Score`

## Claim Information

* `Claim_Type`

The target is created from the fraud label:

```text
Fraud_Flag = Yes → Target = 1
Fraud_Flag = No  → Target = 0
```

---

# ⚠️ Important: Data Leakage Detection

One of the most important parts of this project is the **data leakage analysis**.

During exploration, the following columns were identified as unsuitable for the production feature set:

```text
Fraud_Risk_Score
Fraud_Flag
Claim_Status
Settlement_Amount
```

Identifier columns were also excluded:

```text
Claim_ID
Customer_ID
Policy_Number
```

## Why were these columns removed?

### `Fraud_Flag`

This is the target itself.

Using the target as an input feature would mean giving the model the answer during training.

---

### `Fraud_Risk_Score`

The notebook performs a diagnostic showing that this column can reproduce the fraud label extremely closely.

For example, if a feature is already directly connected to how the fraud label was created, using it can produce unrealistically high model performance.

This is called **target leakage**.

### Why is leakage dangerous?

Suppose a model reports:

```text
Accuracy: 99%
```

This may look excellent.

However, if the model achieved this result because it accidentally used information that would only be known after the fraud decision, the model will fail in a real production environment.

Therefore, the final production model intentionally excludes suspected leakage columns.

> **Important:** The notebook also contains an optional "Legacy-Assisted" experiment using `Fraud_Risk_Score`. This experiment is for comparison and demonstration purposes only and should be used in production only if project or competition rules explicitly confirm that this score is genuinely available before the fraud outcome is known.

---

# ⚙️ Feature Engineering

Feature engineering was used to create additional information from the existing raw data.

The project creates the following engineered features.

---

## 1. Claim to Coverage Ratio

```python
Claim_Coverage_Ratio = Claim_Amount / Coverage_Amount
```

### Why?

This shows how large a claim is relative to the total policy coverage.

### Example

```text
Claim Amount: ₹80,000
Coverage Amount: ₹100,000

Ratio = 80,000 / 100,000 = 0.80
```

A claim consuming a large proportion of available coverage may provide useful information for fraud analysis.

---

## 2. Claim to Premium Ratio

```python
Claim_Premium_Ratio = Claim_Amount / Premium_Amount
```

### Why?

A claim that is disproportionately large compared with the premium may represent a different risk pattern.

### Example

```text
Claim Amount: ₹100,000
Premium Amount: ₹5,000

Ratio = 20
```

---

## 3. Coverage to Premium Ratio

```python
Coverage_Premium_Ratio = Coverage_Amount / Premium_Amount
```

### Why?

This helps represent the relationship between the policy's protection amount and the premium paid.

---

## 4. Remaining Coverage

```python
Remaining_Coverage = Coverage_Amount - Claim_Amount
```

### Why?

This gives the model another representation of how much coverage remains after considering the claim amount.

---

## 5. Young Customer Flag

```python
Young_Customer = 1 if Age < 30
Young_Customer = 0 otherwise
```

### Why?

Tree-based models can learn directly from age, but binary segments can sometimes capture meaningful groups and interactions.

---

## 6. Senior Customer Flag

```python
Senior_Customer = 1 if Age >= 60
Senior_Customer = 0 otherwise
```

---

## 7. Low Credit Score Flag

```python
Low_Credit_Score = 1 if Credit_Score < 600
Low_Credit_Score = 0 otherwise
```

### Why?

This converts a continuous value into a simple risk-related category that may interact differently with other variables.

---

## 8. High Claim Flag

```python
High_Claim = 1 if Claim_Amount >= 100000
High_Claim = 0 otherwise
```

### Why?

Large claims can have a different business impact from smaller claims.

Even when this feature does not directly indicate fraud, it can help the model identify interactions between claim size and other factors.

---

# 🔄 Data Preprocessing

The project uses `ColumnTransformer` and `Pipeline` from Scikit-learn.

## Numeric Features

Numeric columns are handled using:

```python
SimpleImputer(strategy="median")
```

### Why median?

The median is more robust to extreme values than the mean.

For example, claim amounts may contain very large outliers.

```text
1000, 1200, 1500, 2000, 500000
```

The mean would be heavily affected by `500000`, while the median remains more representative of the central value.

---

## Categorical Features

Categorical columns are processed using:

```python
SimpleImputer(strategy="most_frequent")
```

followed by:

```python
OneHotEncoder(handle_unknown="ignore")
```

### Why One-Hot Encoding?

Machine learning models require numerical input.

For example:

```text
Policy_Type:
Health
Vehicle
Home
```

One-hot encoding converts this into:

| Health | Vehicle | Home |
| -----: | ------: | ---: |
|      1 |       0 |    0 |
|      0 |       1 |    0 |
|      0 |       0 |    1 |

### Why `handle_unknown="ignore"`?

A new claim may contain a category that was not present during training.

For example:

```text
Training Regions:
North, South, East

New Region:
Central
```

Using `handle_unknown="ignore"` prevents the prediction pipeline from crashing.

---

# 🧩 Why Scikit-learn Pipeline?

The project combines preprocessing and the machine learning model into a single pipeline.

```text
Raw Claim Data
      ↓
Imputation
      ↓
Encoding
      ↓
Machine Learning Model
      ↓
Prediction
```

This is important because the same preprocessing used during training must also be used when predicting new claims.

Without a pipeline, it is easy to accidentally preprocess training data and production data differently.

---

# 🤖 Models Used

The project evaluates two major machine learning models:

1. Random Forest Classifier
2. XGBoost Classifier

The final production approach is based on **Tuned XGBoost**, according to the model selection workflow implemented in the notebook.

---

# 🌲 Model 1: Random Forest

Random Forest is an ensemble machine learning algorithm that combines many decision trees.

```text
                ┌── Decision Tree 1 ──┐
Input Data ────┼── Decision Tree 2 ──┼── Final Prediction
                ├── Decision Tree 3 ──┤
                └── Decision Tree N ──┘
```

## Why did we use Random Forest?

Random Forest was used as a strong baseline because it:

* Handles non-linear relationships.
* Can model interactions between features.
* Works well with tabular datasets.
* Is relatively robust to noise.
* Can provide feature importance.
* Supports class weighting.

The notebook uses settings such as:

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=16,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)
```

### Why `class_weight="balanced"`?

Fraud datasets are often imbalanced.

For example:

```text
Genuine Claims: 95%
Fraudulent Claims: 5%
```

A model could predict every claim as genuine and still obtain 95% accuracy.

Class balancing gives more importance to the minority fraud class.

---

# ⚡ Model 2: XGBoost

XGBoost is a gradient boosting algorithm that builds trees sequentially.

Each new tree attempts to improve the errors made by previous trees.

```text
Tree 1
   ↓
Learn from errors
   ↓
Tree 2
   ↓
Learn from remaining errors
   ↓
Tree 3
   ↓
Final Strong Model
```

## Why did we use XGBoost?

XGBoost is especially suitable for structured and tabular data.

It was selected because it can:

* Capture complex non-linear relationships.
* Learn feature interactions.
* Use regularization to reduce overfitting.
* Handle imbalanced classification using `scale_pos_weight`.
* Provide feature importance.
* Produce fraud probabilities.
* Work effectively with engineered features.

The project uses an XGBoost configuration similar to:

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.90,
    reg_alpha=0.10,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
```

---

# ⚖️ Why `scale_pos_weight`?

Fraud detection is usually an imbalanced classification problem.

The project calculates:

```python
scale_pos_weight = Number of Genuine Claims / Number of Fraudulent Claims
```

This increases the importance of fraudulent examples during XGBoost training.

### Example

```text
Genuine Claims = 9,000
Fraudulent Claims = 1,000

scale_pos_weight = 9000 / 1000 = 9
```

This helps the model pay greater attention to the minority fraud class.

---

# 🔍 Why `eval_metric="aucpr"`?

The project uses:

```python
eval_metric="aucpr"
```

PR-AUC is particularly useful for imbalanced classification.

Accuracy can be misleading when fraud is rare.

### Example

If 99 out of 100 claims are genuine:

```text
Always predict Genuine
```

The model gets:

```text
Accuracy = 99%
```

But it detects:

```text
Fraud Cases = 0%
```

PR-AUC focuses more directly on how well the model identifies the positive fraud class.

---

# 🧪 Model Validation

The project uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

## Why Stratified K-Fold Cross-Validation?

Stratification ensures that each fold maintains a similar fraud/non-fraud distribution.

The dataset is divided into five parts:

```text
Fold 1 → Validation
Fold 2 → Validation
Fold 3 → Validation
Fold 4 → Validation
Fold 5 → Validation
```

Each part is used once for validation.

This produces **Out-of-Fold (OOF) probabilities**, which provide a more reliable basis for evaluating the model than testing only on the same data used for fitting.

---

# 📊 Evaluation Metrics

The project evaluates the models using several metrics.

## Accuracy

```text
Correct Predictions / Total Predictions
```

Useful as a general metric, but insufficient by itself for fraud detection.

---

## Precision

```text
True Positives / (True Positives + False Positives)
```

Precision answers:

> Of the claims predicted as fraudulent, how many were actually fraudulent?

---

## Recall

```text
True Positives / (True Positives + False Negatives)
```

Recall answers:

> Of all actual fraudulent claims, how many did the model successfully detect?

Recall is especially important when missing a fraudulent claim is costly.

---

## F1 Score

F1 balances precision and recall.

```text
F1 = Harmonic Mean of Precision and Recall
```

---

## F2 Score

The project also calculates the F2 score:

```python
fbeta_score(beta=2)
```

F2 gives more importance to **recall**.

### Why?

In fraud detection, missing a genuine fraud case can be more costly than investigating an additional legitimate claim.

---

## ROC-AUC

ROC-AUC measures how well the model separates the two classes across different thresholds.

---

## PR-AUC

PR-AUC measures the trade-off between precision and recall and is particularly useful for imbalanced fraud datasets.

---

# 🎛️ Hyperparameter Tuning

The XGBoost model is tuned using:

```python
RandomizedSearchCV
```

The search explores parameters such as:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `min_child_weight`
* `subsample`
* `colsample_bytree`
* `reg_alpha`
* `reg_lambda`

The tuning process uses:

```python
n_iter=20
```

and evaluates candidates using:

```python
scoring="average_precision"
```

with stratified cross-validation.

## Why Randomized Search?

Testing every possible combination would be computationally expensive.

Randomized search explores a selected number of parameter combinations more efficiently.

---

# 🎯 Classification Threshold

Machine learning models generate probabilities.

For example:

```text
Claim A → 0.21
Claim B → 0.63
Claim C → 0.91
```

A threshold determines when a claim is classified as fraudulent.

```python
prediction = probability >= threshold
```

The project saves the selected final threshold separately:

```text
claimshield_threshold.pkl
```

This is important because the threshold is part of the final decision system and may not necessarily be the default value of `0.50`.

---

# 🔎 Model Explainability

ClaimShield AI includes two approaches for understanding model behaviour.

## 1. Feature Importance

The project extracts feature importance from the trained XGBoost model.

This helps answer:

> Which features are most influential in the model?

The notebook also visualizes the top 15 important features.

---

## 2. SHAP Explainability

The project uses SHAP:

```python
shap.TreeExplainer
```

SHAP helps explain how features contribute to model predictions.

A SHAP summary plot can help identify:

* Which features are important.
* Whether a feature generally pushes fraud risk up or down.
* How different feature values influence predictions.

This improves transparency and makes the project more useful as a decision-support system.

---

# 💼 Investigator-Friendly Output

The system converts technical predictions into investigator-friendly information.

Example:

```text
Fraud Probability:       0.82
Fraud Risk Score:        82
Fraud Prediction:        Fraudulent
Risk Level:              HIGH
Recommended Action:      INVESTIGATE
Claim Amount:            ₹150,000
Potential Financial Loss: ₹123,000
```

This is more useful to a business user than simply returning:

```text
1
```

---

# 📈 Investigator Dashboard

The project generates a summary of:

* Total claims
* Claims predicted as fraudulent
* Claims to approve
* Claims to review
* Claims to investigate

It also visualizes:

* Recommended action distribution
* Fraud risk score distribution
* Top high-risk claims
* Claims with the highest potential financial loss

---

# 💰 High-Risk Claim Prioritization

Claims can be sorted by:

```text
Fraud Risk Score
```

to identify the most suspicious claims.

They can also be sorted by:

```text
Potential Financial Loss
```

to identify claims with the greatest possible financial impact.

These are not always the same thing.

### Example

```text
Claim A
Fraud Probability: 95%
Claim Amount: ₹10,000
Potential Loss: ₹9,500
```

```text
Claim B
Fraud Probability: 70%
Claim Amount: ₹500,000
Potential Loss: ₹350,000
```

Claim A has a higher fraud probability, but Claim B has a much greater potential financial impact.

This is why ClaimShield AI supports both risk-based and financial prioritization.

---

# 💾 Model Saving

The final system saves three important files.

## Final Model

```text
claimshield_final_xgboost.pkl
```

This contains the full pipeline:

```text
Preprocessing + XGBoost Model
```

---

## Final Threshold

```text
claimshield_threshold.pkl
```

This stores the fraud classification threshold.

---

## Feature List

```text
claimshield_features.pkl
```

This stores the raw production feature names.

The raw feature list is saved instead of one-hot-encoded feature names so that new raw claim data can be aligned correctly before prediction.

---

# 📥 Predicting on New Unlabelled Data

The project supports prediction on an evaluation dataset.

Expected workflow:

```text
evaluation_dataset.csv
        │
        ▼
Feature Engineering
        │
        ▼
Remove Leakage Columns
        │
        ▼
Feature Alignment
        │
        ▼
Load Saved Pipeline
        │
        ▼
Predict Fraud Probability
        │
        ▼
Apply Saved Threshold
        │
        ▼
Generate Business Output
```

The notebook exports the results as:

```text
ClaimShield_AI_Evaluation_Results.csv
```

---

# 🧾 Example Prediction

```python
claim_data = pd.DataFrame([{
    "Age": 35,
    "Gender": "Male",
    "Region": "East",
    "Policy_Type": "Health",
    "Premium_Amount": 12000,
    "Coverage_Amount": 500000,
    "Claim_Amount": 150000,
    "Claim_Type": "Accident",
    "Credit_Score": 620
}])
```

After applying the same feature engineering and prediction pipeline, the output may look like:

```python
{
    "Fraud Probability": 0.73,
    "Fraud Risk Score": 73.0,
    "Fraud Prediction": "Fraudulent",
    "Recommended Action": "INVESTIGATE"
}
```

> The exact prediction will depend on the trained model and saved threshold.

---

# 🛠️ Technologies Used

| Technology       | Purpose                                                |
| ---------------- | ------------------------------------------------------ |
| Python           | Core programming language                              |
| Pandas           | Data loading and manipulation                          |
| NumPy            | Numerical operations                                   |
| Matplotlib       | Data visualization                                     |
| Seaborn          | Statistical visualization                              |
| Scikit-learn     | Preprocessing, pipelines, cross-validation and metrics |
| XGBoost          | Gradient boosting fraud classification model           |
| SHAP             | Model explainability                                   |
| Joblib           | Saving and loading trained models                      |
| Jupyter Notebook | Experimentation and project workflow                   |

---

# 📦 Installation

Clone the project repository:

```bash
git clone <your-repository-url>
cd ClaimShield-AI
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib jupyter
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
ClaimShieldAI_FIXED.ipynb
```

---

# ▶️ How to Run the Project

## Step 1: Place the training dataset

Ensure the dataset is available as:

```text
ClaimShieldAI-Dataset.csv
```

---

## Step 2: Open the notebook

```text
ClaimShieldAI_FIXED.ipynb
```

---

## Step 3: Run the notebook sequentially

The notebook performs:

```text
Import Libraries
      ↓
Load Dataset
      ↓
EDA
      ↓
Target Preparation
      ↓
Leakage Detection
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
Random Forest Training
      ↓
XGBoost Training
      ↓
Cross-Validation
      ↓
XGBoost Hyperparameter Tuning
      ↓
Final Model Evaluation
      ↓
Feature Importance
      ↓
SHAP Explainability
      ↓
Risk Scoring
      ↓
Business Action Recommendation
      ↓
Potential Financial Loss Calculation
      ↓
Save Model
      ↓
Evaluate New Claims
```

---

# 📁 Project Structure

```text
ClaimShield-AI/
│
├── ClaimShieldAI_FIXED.ipynb
├── ClaimShieldAI-Dataset.csv
│
├── evaluation_dataset.csv
│
├── claimshield_final_xgboost.pkl
├── claimshield_threshold.pkl
├── claimshield_features.pkl
│
├── ClaimShield_AI_Evaluation_Results.csv
│
└── README.md
```

---

# 📊 Final Model Results

The notebook automatically prints the final validation metrics for the tuned XGBoost model:

```text
Accuracy
Precision
Recall
F1 Score
F2 Score
ROC-AUC
PR-AUC
Claims Flagged
```

> The exact numerical values should be taken from the final notebook execution because they depend on the dataset and the final tuned model run. They should not be manually invented or hard-coded into the README.

---

# 🔐 Production Data Integrity

The final production feature set is intentionally restricted to:

* Raw customer information
* Policy information
* Claim information
* Financial information
* Engineered features derived from production-safe columns

The following are explicitly excluded:

```text
Fraud_Risk_Score
Fraud_Flag
Claim_Status
Settlement_Amount
Claim_ID
Customer_ID
Policy_Number
```

This separation is important because the goal is to estimate fraud risk using information realistically available when a claim is being evaluated.

---

# ⚠️ Current Limitations

This project is a machine learning prototype and decision-support system.

Some limitations include:

* Model quality depends heavily on the quality and realism of the dataset.
* Historical data may contain bias.
* A predicted fraud risk does not prove that a person committed fraud.
* High-risk predictions should be reviewed by authorized investigators.
* Data distributions can change over time, requiring model monitoring and retraining.
* The leakage diagnostic suggests that some legacy columns may be too closely connected to the target and therefore unsuitable for a realistic production model.

---

# 🔮 Future Improvements

Possible future enhancements include:

* Real-time fraud detection API using Flask or Django.
* Interactive web dashboard.
* Database integration.
* User authentication and investigator roles.
* Individual claim-level SHAP explanations.
* Model monitoring and drift detection.
* Automated retraining pipeline.
* Feedback loop from investigator decisions.
* Cloud deployment.
* Alert system for critical claims.
* Cost-sensitive threshold optimization based on actual investigation costs and fraud losses.

---

# 🏆 Project Outcome

ClaimShield AI demonstrates how machine learning can support the insurance claim investigation process.

The project goes beyond basic classification by combining:

```text
Machine Learning
        +
Feature Engineering
        +
Leakage Detection
        +
Model Validation
        +
Hyperparameter Tuning
        +
Explainable AI
        +
Risk Scoring
        +
Business Decision Rules
        +
Financial Impact Analysis
```

The final result is an **intelligent fraud risk assessment workflow** that can help prioritize insurance claims for approval, review, or investigation.

---

# 👥 Team

**Team Name: HexaRisk**

Project: **ClaimShield AI — Intelligent Insurance Fraud Detection Platform**

Team Size: **6 Members**

---

## ⭐ Conclusion

ClaimShield AI was built not only to predict fraud but to create a practical decision-support workflow for insurance claims.

A major focus of the project is **realistic model development**. The system detects and excludes suspected data leakage, evaluates multiple machine learning models, uses stratified cross-validation, tunes XGBoost, and produces interpretable outputs.

The final system can transform raw claim data into actionable information:

```text
New Claim
    ↓
Fraud Probability
    ↓
Risk Score
    ↓
Fraud Prediction
    ↓
Risk Level
    ↓
Recommended Action
    ↓
Potential Financial Loss
    ↓
Investigator Prioritization
```

**ClaimShield AI: Detect smarter. Investigate faster. Protect better. 🛡️**
