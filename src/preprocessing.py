import os
import pandas as pd
import numpy as np

def _find_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find any of these files: {candidates}")

def load_data():
    sentiment_path = _find_file([
        "data/fear_greed_index.csv",
        "fear_greed_index.csv"
    ])
    trades_path = _find_file([
        "data/historical_data.csv",
        "historical_data.csv"
    ])

    sentiment = pd.read_csv(sentiment_path)
    trades = pd.read_csv(trades_path)

    
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
    sentiment["value"] = pd.to_numeric(sentiment["value"], errors="coerce")
    sentiment = sentiment[["date", "value", "classification"]].dropna()
    sentiment = sentiment.rename(columns={
        "value": "sentiment_value",
        "classification": "sentiment_class"
    })
    sentiment = sentiment.drop_duplicates(subset=["date"])

    
    trades["Timestamp IST"] = pd.to_datetime(trades["Timestamp IST"], errors="coerce")
    trades["trade_date"] = pd.to_datetime(trades["Timestamp IST"].dt.date, errors="coerce")

    numeric_cols = ["Execution Price", "Size Tokens", "Size USD", "Closed PnL", "Fee", "Leverage"]
    for col in numeric_cols:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")

    trades["Closed PnL"] = trades["Closed PnL"].fillna(0)
    trades["Fee"] = trades["Fee"].fillna(0)
    trades["net_pnl"] = trades["Closed PnL"] - trades["Fee"]

    def get_position(x):
        x = str(x).lower()
        if "long" in x:
            return "Long"
        if "short" in x:
            return "Short"
        return "Other"

    trades["position"] = trades["Direction"].apply(get_position) if "Direction" in trades.columns else "Other"

   
    if "Side" in trades.columns:
        trades["Side"] = trades["Side"].astype(str).str.upper().replace({"B": "BUY", "A": "SELL"})
    else:
        trades["Side"] = "UNKNOWN"

    if "Leverage" not in trades.columns:
        trades["Leverage"] = np.nan

    data = pd.merge(
        trades,
        sentiment,
        left_on="trade_date",
        right_on="date",
        how="left"
    )

    data = data.dropna(subset=["sentiment_class", "sentiment_value", "trade_date"]).copy()
    data["win"] = (data["net_pnl"] > 0).astype(int)

    sentiment_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    data["sentiment_class"] = pd.Categorical(
        data["sentiment_class"],
        categories=sentiment_order,
        ordered=True
    )

    return data
