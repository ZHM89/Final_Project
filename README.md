# Medical Cost Analysis - Insight and Prediction

An end-to-end data science project analyzing medical insurance costs and building predictive models to estimate charges based on demographic and lifestyle factors.

## 📌 Project Overview

This project explores the **"Medical Cost Personal Datasets"** from Kaggle, containing 1,338 records with 7 features. The goal is to:
- Perform exploratory data analysis (EDA) to uncover insights
- Conduct statistical tests to validate hypotheses
- Build and compare regression models to predict medical charges
- Optimize the best-performing model through hyperparameter tuning

## 📂 Dataset

- **Source**: [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Size**: 1,338 records × 7 columns
- **Features**:
  - **Categorical**: 
    - `sex`: ['male', 'female']
    - `smoker`: ['yes', 'no'] 
    - `region`: ['southwest', 'southeast', 'northwest', 'northeast']
  - **Numerical**: 
    - `age`: Age of primary beneficiary
    - `bmi`: Body mass index
    - `children`: Number of dependents
- **Target Variable**: 
  - `charges`: Individual medical costs billed by insurance (continuous numeric)

## 🔍 Data Exploration & Inferential Statistics

### EDA Highlights
- **Boxplots** of numerical values to identify distributions and outliers
- **Histograms** showing the right-skewed distribution of charges and the effect of log transformation
- **Boxplots** of charges across categorical features (sex, smoker status, region, age groups, BMI categories)
- **Pairplot & Heatmap** revealing relationships between numerical features

### Key Statistical Findings
1. **T-Test Analysis for Charge Differences Between Smokers & Non-smokers**
   - Hypothesis: Smokers have significantly higher charges
   - Result: T-statistic = 32.75, p-value < 0.0001
   - Conclusion: Validated with $23,615.96 mean difference

2. **ANOVA Test for Charges Across BMI Categories**
   - Hypothesis: Significant differences exist across BMI categories
   - Result: F-statistic = 18.80, p-value < 0.0001
   - Conclusion: Validated - obese individuals have higher charges

3. **Pearson Correlation Between Age and Charges**
   - ρ = 0.30 (p-value < 0.0001) - Moderate positive correlation

4. **Chi-square Test for Smoking and Sex**
   - Chi² = 7.39, p-value = 0.0065
   - Conclusion: Significant association between sex and smoking status

## ⚙️ Preprocessing & Feature Engineering

### Pipeline Summary
- **Target Variable**:
  - Original `charges` and log-transformed `log_charges` to handle skewness
- **Feature Selection**:
  - Used: age, sex, bmi, children, smoker, region
- **Data Splitting**:
  - 80% training & 20% testing (separate splits for original and log-transformed targets)
- **Preprocessing**:
  - Numerical features: Standardized (StandardScaler)
  - Categorical features: One-hot encoded
- **Pipeline Assembly**:
  - Combined preprocessing steps using ColumnTransformer
- **Feature Selection**:
  - SelectKBest with f-regression to evaluate feature relevance

## 🤖 Model Training & Evaluation

### Performance Summary
| Model                | RMSE (Original) | R² (Original) | RMSE (Log) | R² (Log) |
|----------------------|-----------------|---------------|------------|----------|
| Linear Regression    | 5,798.51        | 0.78          | 5,798.51   | 0.78     |
| Ridge Regression     | 5,798.51        | 0.78          | 5,798.51   | 0.78     |
| Lasso Regression     | 5,798.51        | 0.78          | 5,798.51   | 0.78     |
| Random Forest        | 4,518.18        | 0.87          | 4,518.18   | 0.87     |
| **Gradient Boosting**| **4,267.65**    | **0.88**      | **4,267.65**| **0.88** |
| Support Vector       | 8,620.19        | 0.50          | 8,620.19   | 0.50     |

### Best Performing Model
**Gradient Boosting Regressor** with log-transformed target achieved:
- RMSE: $4,267.65
- R²: 0.88

## 🎛 Hyperparameter Tuning

### Optimization Process
- **Objective**: Fine-tune Gradient Boosting with log-transformed target
- **Parameter Grid**:
  - n_estimators: [100, 200]
  - learning_rate: [0.05, 0.1]
  - max_depth: [3, 5]
  - min_samples_split: [2, 5]
  - min_samples_leaf: [1, 2]
- **Method**: 5-fold GridSearchCV
- **Best Parameters**:
  ```python
  {
      'model__learning_rate': 0.1,
      'model__max_depth': 3,
      'model__min_samples_leaf': 2,
      'model__min_samples_split': 5,
      'model__n_estimators': 200
  }