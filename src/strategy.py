import pandas as pd

def run_strategy(data):
    print("\n===== STRATEGY BACKTEST =====")

    def strategy_signal(row):
        if row["sentiment_value"] < 25:
            return "Long"
        elif row["sentiment_value"] > 75:
            return "Short"
        return "None"

    df = data.copy()
    df["signal"] = df.apply(strategy_signal, axis=1)

    aligned = (
        ((df["signal"] == "Long") & (df["position"] == "Long")) |
        ((df["signal"] == "Short") & (df["position"] == "Short"))
    )

    strategy_df = df[aligned].copy()
    strategy_pnl = round(strategy_df["net_pnl"].sum(), 2)
    total_pnl = round(df["net_pnl"].sum(), 2)
    contribution_pct = round((strategy_pnl / total_pnl) * 100, 2) if total_pnl != 0 else 0.0
    strategy_trades = int(strategy_df.shape[0])

    summary = pd.DataFrame([{
        "strategy_trades": strategy_trades,
        "strategy_pnl": strategy_pnl,
        "total_pnl": total_pnl,
        "contribution_pct": contribution_pct
    }])
    summary.to_csv("outputs/strategy_summary.csv", index=False)

    print("Strategy Trades:", strategy_trades)
    print("Strategy PnL:", strategy_pnl)
    print("Total PnL:", total_pnl)
    print("Strategy Contribution %:", contribution_pct)

    return {
        "strategy_trades": strategy_trades,
        "strategy_pnl": strategy_pnl,
        "total_pnl": total_pnl,
        "contribution_pct": contribution_pct
    }
