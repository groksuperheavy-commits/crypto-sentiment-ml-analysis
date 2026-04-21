import matplotlib
matplotlib.use("Agg")
import os
import pandas as pd
import matplotlib.pyplot as plt

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def run_analysis(data):
    results = {}

    perf = data.groupby("sentiment_class", observed=False)["net_pnl"].agg(["sum", "mean", "count"]).round(2)
    win_rate = data.groupby("sentiment_class", observed=False)["win"].mean().mul(100).round(2)
    position_counts = pd.crosstab(data["sentiment_class"], data["position"])
    avg_trade_size = data.groupby("sentiment_class", observed=False)["Size USD"].mean().round(2)

    daily = data.groupby("trade_date").agg({
        "net_pnl": "sum",
        "sentiment_value": "mean"
    }).dropna()
    corr = round(daily["net_pnl"].corr(daily["sentiment_value"]), 4)

    trader_perf = data.groupby("Account").agg(
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        trades=("net_pnl", "count"),
        total_fees=("Fee", "sum")
    ).sort_values("total_pnl", ascending=False).round(2)

    coin_perf = data.groupby("Coin").agg(
        total_pnl=("net_pnl", "sum"),
        trades=("net_pnl", "count"),
        avg_pnl=("net_pnl", "mean")
    ).sort_values("total_pnl", ascending=False).round(2)

    results["perf"] = perf
    results["win_rate"] = win_rate
    results["position_counts"] = position_counts
    results["avg_trade_size"] = avg_trade_size
    results["corr"] = corr
    results["trader_perf"] = trader_perf
    results["coin_perf"] = coin_perf

    # Save tables
    perf.to_csv("outputs/performance_by_sentiment.csv")
    win_rate.to_csv("outputs/win_rate_by_sentiment.csv")
    position_counts.to_csv("outputs/position_counts.csv")
    avg_trade_size.to_csv("outputs/avg_trade_size_by_sentiment.csv")
    trader_perf.head(15).to_csv("outputs/top_traders.csv")
    coin_perf.head(15).to_csv("outputs/top_coins.csv")
    daily.to_csv("outputs/daily_sentiment_pnl.csv")

    # Plot 1: Avg PnL by sentiment
    plt.figure(figsize=(8, 5))
    perf["mean"].plot(kind="bar")
    plt.title("Average Net PnL by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Net PnL")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/avg_pnl_by_sentiment.png")
    plt.close()

    # Plot 2: Win rate by sentiment
    plt.figure(figsize=(8, 5))
    win_rate.plot(kind="bar")
    plt.title("Win Rate by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Win Rate (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/win_rate_by_sentiment.png")
    plt.close()

    # Plot 3: Long vs Short by sentiment
    if {"Long", "Short"}.issubset(position_counts.columns):
        plt.figure(figsize=(9, 5))
        position_counts[["Long", "Short"]].plot(kind="bar", stacked=False)
        plt.title("Long vs Short Position Counts by Sentiment")
        plt.xlabel("Sentiment")
        plt.ylabel("Trade Count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/long_short_by_sentiment.png")
        plt.close()

    # Plot 4: Trade size by sentiment
    plt.figure(figsize=(8, 5))
    avg_trade_size.plot(kind="bar")
    plt.title("Average Trade Size (USD) by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Size USD")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/avg_trade_size_by_sentiment.png")
    plt.close()

    # Plot 5: Sentiment vs daily pnl scatter
    plt.figure(figsize=(8, 5))
    plt.scatter(daily["sentiment_value"], daily["net_pnl"])
    plt.title(f"Daily Sentiment vs Daily Net PnL (corr={corr})")
    plt.xlabel("Sentiment Value")
    plt.ylabel("Daily Net PnL")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/sentiment_vs_daily_pnl.png")
    plt.close()

    # Plot 6: Top coins
    plt.figure(figsize=(9, 6))
    coin_perf.head(10)["total_pnl"].sort_values().plot(kind="barh")
    plt.title("Top 10 Coins by Total Net PnL")
    plt.xlabel("Total Net PnL")
    plt.ylabel("Coin")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/top_coins_by_pnl.png")
    plt.close()

    # Plot 7: Top traders
    plt.figure(figsize=(9, 6))
    trader_perf.head(10)["total_pnl"].sort_values().plot(kind="barh")
    plt.title("Top 10 Traders by Total Net PnL")
    plt.xlabel("Total Net PnL")
    plt.ylabel("Account")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/top_traders_by_pnl.png")
    plt.close()

    print("\n===== ANALYSIS =====")
    print("\nPerformance by Sentiment:\n", perf)
    print("\nWin Rate by Sentiment (%):\n", win_rate)
    print("\nCorrelation (Daily Sentiment vs Daily Net PnL):", corr)
    print("\nTop 5 Traders:\n", trader_perf.head())
    print("\nTop 5 Coins:\n", coin_perf.head())

    return results
