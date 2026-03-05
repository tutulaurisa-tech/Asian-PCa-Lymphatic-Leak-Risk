"""
Script 01: Statistical Analysis and Risk Factor Identification
Project: The Ethnic Rift in Metabolic Risk (Asian PCa Chyle Leak Study)
Author: [Your Name/Research Group]
Date: 2026-03-05
Description: This script performs baseline characteristic comparison, univariate, 
             and multivariate logistic regression analysis to identify predictors of Chyle Leak.
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
import statsmodels.api as sm

# ---------------------------------------------------------
# 1. Data Loading and Preprocessing
# ---------------------------------------------------------
# Load the dataset (Ensure your CSV file is in the same directory)
# Note: Real patient data is not included for privacy reasons.
df = pd.read_csv('data.csv') 

# Define the outcome variable
outcome_col = 'Chyle_Leak' # 1 = Yes, 0 = No

# Continuous and Categorical variables definition
continuous_vars = ['Age', 'BMI', 'PSA', 'Lymph_Node_Yield', 'Operative_Time', 'Blood_Loss']
categorical_vars = ['NCCN_Risk_Group', 'Surgical_Approach', 'Surgical_Modality', 'Nerve_Sparing']

# ---------------------------------------------------------
# 2. Baseline Characteristics (Table 1)
# ---------------------------------------------------------
print("--- Baseline Characteristics Comparison ---")
for var in continuous_vars:
    group0 = df[df[outcome_col] == 0][var].dropna()
    group1 = df[df[outcome_col] == 1][var].dropna()
    stat, p = mannwhitneyu(group0, group1)
    print(f"{var}: Median (No Leak)={group0.median()}, Median (Leak)={group1.median()}, P-value={p:.4f}")

for var in categorical_vars:
    contingency_table = pd.crosstab(df[var], df[outcome_col])
    # Use Fisher's exact test if small sample size, else Chi-square
    if contingency_table.min().min() < 5:
        stat, p = fisher_exact(contingency_table)
    else:
        stat, p, dof, expected = chi2_contingency(contingency_table)
    print(f"{var}: P-value={p:.4f}")

# ---------------------------------------------------------
# 3. Univariate Logistic Regression (Table 2)
# ---------------------------------------------------------
print("\n--- Univariate Logistic Regression ---")
univ_results = []
for var in continuous_vars + categorical_vars:
    X = df[[var]].dropna()
    # Handle categorical variables (dummy encoding if needed)
    if df[var].dtype == 'object':
        X = pd.get_dummies(df[var], drop_first=True)
    
    y = df.loc[X.index, outcome_col]
    X = sm.add_constant(X)
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
        odds_ratio = np.exp(model.params.iloc[1])
        p_value = model.pvalues.iloc[1]
        conf = model.conf_int().iloc[1]
        print(f"{var}: OR={odds_ratio:.2f}, 95% CI [{np.exp(conf[0]):.2f}-{np.exp(conf[1]):.2f}], P={p_value:.4f}")
        
        # Select variables with P < 0.1 for multivariate analysis
        if p_value < 0.1:
            univ_results.append(var)
    except:
        print(f"Error fitting model for {var}")

# ---------------------------------------------------------
# 4. Multivariate Logistic Regression (Table 3)
# ---------------------------------------------------------
print("\n--- Multivariate Logistic Regression ---")
# Manually selecting significant variables based on Univariate results + Clinical relevance
multivar_features = ['BMI', 'Lymph_Node_Yield', 'Operative_Time', 'Surgical_Modality'] 

X_multi = df[multivar_features].dropna()
# Ensure data alignment
y_multi = df.loc[X_multi.index, outcome_col]

# Convert categorical to dummy if necessary
X_multi = pd.get_dummies(X_multi, drop_first=True)
X_multi = sm.add_constant(X_multi)

model_multi = sm.Logit(y_multi, X_multi).fit()
print(model_multi.summary())

# Extract adjusted ORs
params = model_multi.params
conf = model_multi.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print("\nAdjusted Odds Ratios:")
print(np.exp(conf))