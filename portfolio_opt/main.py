from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers=tickers, start=start_date, end=end_date)["Adj Close"]
    return data


tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NFLX", "NVDA"]

start_date = "2020-01-01"
end_date = "2024-01-01"

data = get_stock_data(tickers, start_date, end_date)

returns = data.pct_change().dropna()
mean_return = returns.mean()
covariance = returns.cov()


def portfolio_performance(weights, mean_returns, covariance):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(252)
    return returns, std_dev


def neg_sharpe_ratio(weights, mean_returns, covariance, risk_free_rate=0.01):
    portfolio_returns, portfolio_std_dev = portfolio_performance(weights, mean_returns, covariance)
    return - (portfolio_returns - risk_free_rate) / portfolio_std_dev


def optimal_portfolio(mean_returns, covariance, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    args = (mean_returns, covariance, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0, 1)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio,
                      num_assets * [1./num_assets,],
                      args=args,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints
                      )
    return result


result = optimal_portfolio(mean_return, covariance)
optimal_weights = result.x
print(optimal_weights)


def plot_efficient_frontier(mean_returns, covariance, num_portfolios=10000, risk_free_rate=0.01):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.rand(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_std_dev = portfolio_performance(weights, mean_returns, covariance)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

    max_sharpe = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe], results[1, max_sharpe]
    max_sharpe_allocation = weights_record[max_sharpe]

    plt.figure(figsize=(10, 8))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap="YlGnBu", marker='o')
    plt.colorbar(label='Sharpe ratio')
    plt.scatter(sdp, rp, c='red', marker='*', s=500, label="Maximum Sharpe ratio")
    plt.title("Portfolio Optimization based on Efficient Frontier")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Portfolio Return")
    plt.legend(labelspacing=0.8)
    plt.show()


plot_efficient_frontier(mean_return, covariance)
