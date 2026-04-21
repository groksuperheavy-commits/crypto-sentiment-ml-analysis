def generate_insights(analysis_results, model_results, strategy_results):
    perf = analysis_results["perf"]
    win_rate = analysis_results["win_rate"]
    avg_trade_size = analysis_results["avg_trade_size"]
    position_counts = analysis_results["position_counts"]
    corr = analysis_results["corr"]
    trader_perf = analysis_results["trader_perf"]
    coin_perf = analysis_results["coin_perf"]

    best_win_sentiment = win_rate.idxmax() if len(win_rate) else "N/A"
    worst_perf_sentiment = perf["mean"].idxmin() if len(perf) else "N/A"
    top_trader = trader_perf.index[0] if len(trader_perf) else "N/A"
    top_coin = coin_perf.index[0] if len(coin_perf) else "N/A"

    with open("outputs/insights.txt", "w", encoding="utf-8") as f:
        f.write("=== FINAL INSIGHTS REPORT ===\n\n")
        f.write("1. Market Regime Performance\n")
        f.write(str(perf))
        f.write("\n\n2. Win Rate by Sentiment (%)\n")
        f.write(str(win_rate))
        f.write("\n\n3. Position Counts\n")
        f.write(str(position_counts))
        f.write("\n\n4. Average Trade Size (USD)\n")
        f.write(str(avg_trade_size))
        f.write("\n\n5. Daily Sentiment-PnL Correlation\n")
        f.write(str(corr))
        f.write("\n\n6. Model Metrics\n")
        f.write(f"Accuracy: {model_results['accuracy']}\n")
        f.write(f"ROC-AUC: {model_results['roc_auc']}\n")
        f.write("\n7. Strategy Metrics\n")
        f.write(f"Strategy Trades: {strategy_results['strategy_trades']}\n")
        f.write(f"Strategy PnL: {strategy_results['strategy_pnl']}\n")
        f.write(f"Total PnL: {strategy_results['total_pnl']}\n")
        f.write(f"Strategy Contribution %: {strategy_results['contribution_pct']}\n")

        f.write("\n\n=== INTERPRETED INSIGHTS ===\n")
        f.write(f"- Best win-rate regime: {best_win_sentiment}.\n")
        f.write(f"- Weak daily sentiment-to-PnL correlation ({corr}) suggests sentiment level alone is not enough.\n")
        f.write(f"- Worst average-PnL regime: {worst_perf_sentiment}.\n")
        f.write(f"- Top trader by net PnL: {top_trader}.\n")
        f.write(f"- Top coin by net PnL: {top_coin}.\n")
        f.write("- Trade size by sentiment helps measure trader conviction.\n")
        f.write("- Strategy logic uses a contrarian rule: Long in fear, Short in greed.\n")
        f.write("- Model performance should be judged with ROC-AUC as well, not accuracy alone.\n")
