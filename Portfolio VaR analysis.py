import sqlite3
import matplotlib.pyplot as plt
import numpy as np

def get_ftse100_companies():
    connection = sqlite3.connect('ftse100yr_to_date.db')
    cursor = connection.cursor()

    query = "SELECT id, name, code FROM stock_names"
    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    ftse100_companies = []
    stock_index_map = {}
    for row in results:
        stock_id, name, code = row
        ftse100_companies.append({'id': stock_id, 'name': name, 'code': code})
        stock_index_map[code] = stock_id
    
    return ftse100_companies, stock_index_map

def get_stock_prices(stock_id):
    connection = sqlite3.connect('ftse100yr_to_date.db')
    cursor = connection.cursor()

    query = "SELECT closing_price FROM stock_prices WHERE stock_id = ?"
    cursor.execute(query, (stock_id,))
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    prices = [price[0] for price in results]
    return prices

def calculate_daily_returns(prices):
    daily_returns = []
    for i in range(1, len(prices)):
        prev_close = prices[i - 1]
        curr_close = prices[i]
        daily_return = ((curr_close - prev_close) / prev_close)
        daily_returns.append(daily_return)
    return daily_returns

print("Welcome to the FTSE100 Stock Portfolio VaR Tester")

while True:
    confidence_level = float(input("Confidence level (in decimals between 0.75 and 1): "))
    if 0.75 < confidence_level < 1:
        break
    print("Please enter a confidence level between 0.75 and 1.")

ftse100_companies, stock_index_map = get_ftse100_companies()

print("Here are all the FTSE 100 companies and their stock tickers:\n")

for company in ftse100_companies:
    print(f"{company['name']} ({company['code']})")

while True:
    num_companies = int(input("\nNumber of companies in the portfolio: "))
    if 1 <= num_companies <= 100:
        break
    print("Portfolio must be between 1 and 100 stocks.")

stock_tickers = []
investment_amounts = []

for i in range(1, num_companies + 1):
    while True:
        ticker = input(f"Stock {i}: ").strip().upper()
        if ticker and ticker in stock_index_map:
            break
        else:
            print(f"Ticker '{ticker}' is not valid. Please enter a valid ticker (must be in caps).")

    while True:
        try:
            investment_amount = float(input(f"Investment amount for {ticker}: "))
            break
        except ValueError:
            print("Please enter a valid number for the investment amount.")

    stock_tickers.append(ticker)
    investment_amounts.append(investment_amount)

stock_data = {}
min_length = None
for ticker in stock_tickers:
    stock_id = stock_index_map[ticker]
    prices = get_stock_prices(stock_id)
    stock_data[ticker] = prices
    if min_length is None or len(prices) < min_length:
        min_length = len(prices)

for ticker in stock_data:
    stock_data[ticker] = stock_data[ticker][:min_length]
    
stock_returns = []
for ticker in stock_tickers:
    prices = stock_data[ticker]
    daily_returns = calculate_daily_returns(prices)
    stock_returns.append(daily_returns)

def historical_VaR():
    returns_matrix = np.array(stock_returns)

    weights = np.array(investment_amounts)
    portfolio_returns = np.dot(returns_matrix.T, weights)

    sorted_returns = np.sort(portfolio_returns)

    var_index = int((1 - confidence_level) * len(sorted_returns))
    var_value = sorted_returns[var_index]

    total_investment = sum(investment_amounts)
    var_value = round(var_value, 2)
    
    print(f'\n--- Historical Daily VaR ---')
    print(f'Total investment = £{total_investment}')
    print(f"Historical VaR at {confidence_level * 100}% confidence level: £{abs(var_value)}")

    return sorted_returns, var_value, total_investment

def monte_carlo_VaR():
    returns_matrix = np.array(stock_returns)

    if len(stock_tickers) == 1:
        mean_returns = np.mean(returns_matrix)
        std_dev_returns = np.std(returns_matrix)

        num_simulations = 10000

        simulated_returns = np.random.normal(mean_returns, std_dev_returns, num_simulations)

        total_investment = investment_amounts[0]
        profit_loss = total_investment * simulated_returns

    else:
        returns_matrix = returns_matrix.T

        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix, rowvar=False)

        num_simulations = 10000

        try:
            chol_matrix = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-10
            chol_matrix = np.linalg.cholesky(cov_matrix)

        z = np.random.normal(size=(num_simulations, len(stock_tickers)))

        correlated_random_returns = z @ chol_matrix.T

        simulated_returns = correlated_random_returns + mean_returns

        total_investment = sum(investment_amounts)
        weights = np.array(investment_amounts) / total_investment

        portfolio_returns = simulated_returns @ weights

        profit_loss = total_investment * portfolio_returns

    profit_loss_sorted = np.sort(profit_loss)

    var_index = int((1 - confidence_level) * num_simulations)
    var_value = profit_loss_sorted[var_index]
    var_value = round(var_value, 2)

    print(f'\n--- Monte Carlo Daily VaR ---')
    print(f'Total investment = £{total_investment}')
    print(f"Monte Carlo VaR at {confidence_level * 100}% confidence level: £{abs(var_value)}")

    return profit_loss, var_value, total_investment

def plot_VaR_results(sorted_returns, hist_var_value, profit_loss, mc_var_value, total_investment):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].hist(sorted_returns, bins=50, edgecolor='black')
    axs[0].axvline(hist_var_value, color='red', linestyle='dashed', linewidth=2)
    axs[0].set_title(f'Historical VaR (Daily)\nTotal investment = £{total_investment}\nVaR = £{abs(hist_var_value)}', fontsize=12)
    axs[0].set_xlabel('Daily Profit/Loss (£)')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True)

    axs[1].hist(profit_loss, bins=50, edgecolor='black')
    axs[1].axvline(mc_var_value, color='red', linestyle='dashed', linewidth=2)
    axs[1].set_title(f'Monte Carlo VaR (Daily)\nTotal investment = £{total_investment}\nVaR = £{abs(mc_var_value)}', fontsize=12)
    axs[1].set_xlabel('Daily Profit/Loss (£)')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

sorted_returns, hist_var_value, total_investment = historical_VaR()
profit_loss, mc_var_value, _ = monte_carlo_VaR()

plot_VaR_results(sorted_returns, hist_var_value, profit_loss, mc_var_value, total_investment)
