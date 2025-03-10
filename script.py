import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  # For feature scaling

# --- 1. Load and Investigate the Data ---
df = pd.read_csv('tennis_stats.csv')
print(df.head())  # Display the first few rows
print(df.info())  # Display data types and non-null counts
print(df[df['Wins'] == 48])  # Display rows where Wins equals 48

# --- 2. Exploratory Data Analysis (EDA) ---

# 2.1 Scatter Plot: Wins vs. Winnings
x_wins_eda = df['Wins']
y_winnings_eda = df['Winnings']
plt.scatter(x_wins_eda, y_winnings_eda)
plt.xlabel("Wins")
plt.ylabel("Winnings")
plt.title("Wins vs. Winnings")
plt.show()
plt.clf()  # Clear the plot

# 2.2 Scatter Plot: Wins vs. Ranking
y_ranking_eda = df['Ranking']
plt.scatter(x_wins_eda, y_ranking_eda)
plt.xlabel("Wins")
plt.ylabel("Ranking")
plt.title("Wins vs. Ranking")
plt.show()
plt.clf()

# 2.3 Scatter Plot: BreakPointsOpportunities vs. Winnings
x_bpo_eda = df['BreakPointsOpportunities']
plt.scatter(x_bpo_eda, y_winnings_eda)
plt.xlabel("Break Points Opportunities")
plt.ylabel("Winnings")
plt.title("BPO vs. Winnings")
plt.show()
plt.clf()

# --- 3. Single Feature Linear Regression ---

# 3.1 BreakPointsOpportunities vs. Winnings
x_bpo_single = df[['BreakPointsOpportunities']]  # DataFrame for sklearn
y_winnings_single = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(x_bpo_single, y_winnings_single, test_size=0.2, random_state=1)

model_bpo = LinearRegression()
model_bpo.fit(x_train, y_train)

print(f"BreakPointsOpportunities vs. Winnings - Score: {model_bpo.score(x_test, y_test)}")
print(f"BreakPointsOpportunities vs. Winnings - Coefficient: {model_bpo.coef_}")
print(f"BreakPointsOpportunities vs. Winnings - Intercept: {model_bpo.intercept_}")

# Plotting predictions vs. actual for BreakPointsOpportunities vs. Winnings
y_pred_bpo = model_bpo.predict(x_test)
plt.scatter(x_test, y_test, color='blue', label='Actual Data')
plt.plot(x_test, y_pred_bpo, color='red', label='Predicted Data')  # Regression line
plt.xlabel('Break Points Opportunities')
plt.ylabel('Winnings')
plt.title('Break Points Opportunities vs. Winnings (Actual vs. Predicted)')
plt.legend()
plt.show()
plt.clf()

# 3.2 Wins vs. Ranking
x_wins_single = df[['Wins']]
y_ranking_single = df['Ranking']

x_train, x_test, y_train, y_test = train_test_split(x_wins_single, y_ranking_single, test_size=0.2, random_state=1)

model_wins_rank = LinearRegression()
model_wins_rank.fit(x_train, y_train)

y_pred_wins_rank = model_wins_rank.predict(x_test)

plt.scatter(x_test, y_test, color='blue', label='Actual Data')
plt.plot(x_test, y_pred_wins_rank, color='red', label='Predicted Data')
plt.xlabel('Wins')
plt.ylabel('Ranking')
plt.title('Wins vs. Ranking (Actual vs. Predicted)')
plt.legend()
plt.show()
plt.clf()

print(f"Wins vs. Ranking - Score: {model_wins_rank.score(x_test, y_test)}")

# --- 4. Two Feature Linear Regression ---

# 4.1 Wins and BreakPointsOpportunities vs. Winnings
features_two = ['Wins', 'BreakPointsOpportunities']
x_features_two = df[features_two]
y_winnings_two = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(x_features_two, y_winnings_two, test_size=0.2, random_state=1)

model_two = LinearRegression()
model_two.fit(x_train, y_train)

print(f"Wins and BreakPointsOpportunities vs. Winnings - Score: {model_two.score(x_test, y_test)}")
print(f"Wins and BreakPointsOpportunities vs. Winnings - Coefficients: {model_two.coef_}")
print(f"Wins and BreakPointsOpportunities vs. Winnings - Intercept: {model_two.intercept_}")

# --- 5. Multiple Feature Linear Regression ---

# 5.1 All (or selected) features vs. Winnings
features_all = ['Year', 'FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon',
                 'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces',
                 'BreakPointsConverted', 'BreakPointsFaced', 'BreakPointsOpportunities',
                 'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon',
                 'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon',
                 'TotalPointsWon', 'TotalServicePointsWon', 'Wins', 'Losses', 'Ranking']
x_all = df[features_all]
y_winnings_all = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(x_all, y_winnings_all, test_size=0.2, random_state=1)

# Feature Scaling (Important for Multiple Linear Regression)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model_all = LinearRegression()
model_all.fit(x_train_scaled, y_train)

y_pred_all = model_all.predict(x_test_scaled)

print(f"All Features vs. Winnings - Score: {model_all.score(x_test_scaled, y_test)}")
print(f"All Features vs. Winnings - Coefficients: {model_all.coef_}")
print(f"All Features vs. Winnings - Intercept: {model_all.intercept_}")

# Visualize Predictions vs Actual
plt.scatter(y_test, y_pred_all)
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.title("Actual Winnings vs. Predicted Winnings (Multiple Features)")
plt.show()

# Residual plot
residuals = y_test - y_pred_all
plt.scatter(y_pred_all, residuals)
plt.xlabel('Predicted Winnings')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
