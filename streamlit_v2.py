import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (12, 6)

@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    df_clean = df.copy()
    cat_cols = ['sex', 'smoker', 'region']
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype('category')
    df_clean['log_charges'] = np.log1p(df_clean['charges'])
    df_clean['age_group'] = pd.cut(df_clean['age'], 
                                 bins=[0, 18, 30, 45, 60, 100],
                                 labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    df_clean['bmi_category'] = pd.cut(df_clean['bmi'],
                                     bins=[0, 18.5, 25, 30, 100],
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    return df_clean

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", 
                        ["Data Overview", "Exploratory Analysis", 
                         "Statistical Tests", "Model Performance"])

st.title("Medical Insurance Cost Analysis")

if page == "Data Overview":
    st.header("Dataset Overview")
    st.subheader("First 5 Rows")
    st.dataframe(df.head())
    st.subheader("Dataset Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())
    st.subheader("Categorical Variables")
    st.write("Sex categories:", df['sex'].unique())
    st.write("Smoker categories:", df['smoker'].unique())
    st.write("Region categories:", df['region'].unique())

elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    st.subheader("Distribution of Insurance Charges")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df['charges'], bins=50, kde=True, ax=ax1)
    ax1.set_title('Original Charges')
    ax1.set_xlabel('Charges ($)')
    sns.histplot(df['log_charges'], bins=50, kde=True, ax=ax2)
    ax2.set_title('Log-Transformed Charges')
    ax2.set_xlabel('Log(Charges)')
    st.pyplot(fig)
    st.subheader("Charges by Categorical Features")
    cat_feature = st.selectbox("Select categorical feature", 
                              ['sex', 'smoker', 'region', 'age_group', 'bmi_category'])
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x=cat_feature, y='charges', data=df)
    plt.title(f'Charges by {cat_feature}')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.subheader("Correlation Heatmap")
    num_cols = ['age', 'bmi', 'children', 'charges']
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0)
    st.pyplot(fig)
    st.subheader("3D Relationship: Age, BMI, and Charges")
    fig = px.scatter_3d(df, x='age', y='bmi', z='charges',
                        color='smoker', size='children',
                        hover_data=['sex', 'region'],
                        title='3D Relationship: Age, BMI, and Charges',
                        width=800, height=600)
    st.plotly_chart(fig)

elif page == "Statistical Tests":
    st.header("Statistical Analysis")
    st.subheader("Smokers vs Non-Smokers Charges Comparison")
    smoker_charges = df[df['smoker'] == 'yes']['charges']
    non_smoker_charges = df[df['smoker'] == 'no']['charges']
    t_stat, p_value = stats.ttest_ind(smoker_charges, non_smoker_charges, equal_var=False)
    st.write(f"T-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        mean_diff = smoker_charges.mean() - non_smoker_charges.mean()
        st.write(f"**Conclusion:** Smokers have significantly higher charges (${mean_diff:,.2f} difference).")
    else:
        st.write("**Conclusion:** No significant difference in charges between smokers and non-smokers.")
    st.subheader("Charges Across BMI Categories")
    bmi_groups = df['bmi_category'].unique()
    charges_by_bmi = [df[df['bmi_category'] == bmi]['charges'] for bmi in bmi_groups]
    f_stat, p_value = stats.f_oneway(*charges_by_bmi)
    st.write(f"F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        st.write("**Conclusion:** There are significant differences in charges across BMI categories.")
    else:
        st.write("**Conclusion:** No significant differences in charges across BMI categories.")
    st.subheader("Age and Charges Correlation")
    pearson_corr, pearson_p = stats.pearsonr(df['age'], df['charges'])
    st.write(f"Pearson Correlation: {pearson_corr:.2f} (p-value: {pearson_p:.4f})")
    st.subheader("Association Between Sex and Smoking Status")
    contingency_table = pd.crosstab(df['sex'], df['smoker'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    st.write(f"Chi2-statistic: {chi2:.2f}, p-value: {p:.4f}")
    if p < 0.05:
        st.write("**Conclusion:** There is a significant association between sex and smoking status.")
    else:
        st.write("**Conclusion:** No significant association between sex and smoking status.")

elif page == "Model Performance":
    st.header("Machine Learning Model Performance")
    features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    target = 'charges'
    log_target = 'log_charges'
    X = df[features]
    y = df[target]
    y_log = df[log_target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, _, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    numeric_features = ['age', 'bmi', 'children']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_features = ['sex', 'smoker', 'region']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }
    results = []
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            'Model': name,
            'RMSE': f"{rmse:,.2f}",
            'MAE': f"{mae:,.2f}",
            'R2': f"{r2:.3f}"
        })
    results_df = pd.DataFrame(results)
    st.subheader("Model Performance Summary")
    st.dataframe(results_df)

    st.subheader("Performance Analysis")
    best_model = results_df.sort_values(by="RMSE").iloc[0]
    st.write(f"Best performing model: **{best_model['Model']}** with RMSE = {best_model['RMSE']}, MAE = {best_model['MAE']}, and R² = {best_model['R2']}")
