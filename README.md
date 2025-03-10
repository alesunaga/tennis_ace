# Tennis Statistics Analysis

This Python script performs an analysis of tennis player statistics, including exploratory data analysis and linear regression modeling.

## 1. Data Loading and Investigation

The script starts by loading the `tennis_stats.csv` dataset using Pandas. It then displays the first few rows, data information, and specific rows where the 'Wins' column equals 48.

## 2. Exploratory Data Analysis (EDA)

Scatter plots are generated to visualize the relationships between:

-   'Wins' and 'Winnings'
-   'Wins' and 'Ranking'
-   'BreakPointsOpportunities' and 'Winnings'

## 3. Single Feature Linear Regression

Linear regression models are trained using single features to predict 'Winnings' and 'Ranking'.

-   'BreakPointsOpportunities' vs. 'Winnings'
-   'Wins' vs. 'Ranking'

The models' scores, coefficients, and intercepts are printed. Scatter plots of actual vs. predicted values are also generated.

## 4. Two Feature Linear Regression

A linear regression model is trained using 'Wins' and 'BreakPointsOpportunities' to predict 'Winnings'.

## 5. Multiple Feature Linear Regression

A linear regression model is trained using all (or selected) features to predict 'Winnings'.

-   Feature scaling using `StandardScaler` is applied.
-   The model's scores, coefficients, and intercepts
