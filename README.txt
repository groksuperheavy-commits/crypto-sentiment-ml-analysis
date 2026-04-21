# Crypto Sentiment vs Trader Performance Analysis

This project studies how **Bitcoin market sentiment** relates to **Hyperliquid trader performance** using two datasets:

- Bitcoin Fear & Greed Index
- Historical Hyperliquid trade data

The objective is to uncover behavioral patterns, evaluate whether sentiment helps explain trading outcomes, and test whether a simple sentiment-aware strategy can generate useful signals.



## Problem Statement

This project answers questions such as:

- How does trader profitability change across **Fear**, **Neutral**, and **Greed** market regimes?
- Do traders behave differently during extreme sentiment periods?
- Can sentiment be used as a useful feature in a predictive model?
- Can a simple sentiment-based trading strategy capture meaningful profit?



## Datasets Used

### 1. Bitcoin Market Sentiment Dataset
Important columns:
- `date`
- `value`
- `classification`

### 2. Historical Trader Data from Hyperliquid
Important columns:
- `Account`
- `Coin`
- `Execution Price`
- `Size Tokens`
- `Size USD`
- `Side`
- `Timestamp IST`
- `Direction`
- `Closed PnL`
- `Fee`


## Project Workflow


Load Data
→ Clean and preprocess both datasets
→ Engineer features (trade_date, net_pnl, win, position)
→ Merge sentiment data with trade data on date
→ Perform exploratory analysis
→ Train ML model
→ Backtest sentiment-based strategy
→ Generate final insights report


## Project Structure


ultimate_crypto_project_v2/
│
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── fear_greed_index.csv
│   └── historical_data.csv
│
├── src/
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── model.py
│   ├── strategy.py
│   └── insights.py
│
└── outputs/
    ├── insights.txt
    ├── model_report.txt
    ├── performance_by_sentiment.csv
    ├── win_rate_by_sentiment.csv
    ├── top_traders.csv
    ├── top_coins.csv
    ├── feature_importance_top20.csv
    ├── strategy_summary.csv
    └── plots/





## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn





## Author

**Niharika Chauhan**  
B.Tech CSE (AI)  
Machine Learning / AI / Data Analytics Enthusiast
