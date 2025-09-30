# ----------------------------------- Libraries -----------------------------------
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# Loads historical stock data and engineers features for model training.
# ticker: Stock ticker symbol (e.g. "AAPL", "MSFT", "TSLA", "^GSPC")
# period: Data period to download (default "max" for maximum available data)
def load_and_engineer_data(ticker, period="max"):
    # Download stock data
    df = yf.Ticker(ticker).history(period=period)
    
    # Check if data was retrieved
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the ticker symbol.")
    
    # Clean up columns
    if 'Dividends' in df.columns:
        del df['Dividends']
    if 'Stock Splits' in df.columns:
        del df['Stock Splits']
    
    # Create target variables
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    
    # Filter data from 1990 onwards
    df = df.loc["1990-01-01":].copy()
    
    # Check if we have enough data after filtering
    if len(df) < 50:
        raise ValueError(f"Not enough historical data for ticker '{ticker}'. Need at least 50 days, got {len(df)}.")
    
    # Engineer features
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []
    for horizon in horizons:
        # Skip horizons that are longer than our data
        if horizon >= len(df):
            continue
            
        rolling_averages = df.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
        new_predictors += [ratio_column, trend_column]
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Final check for sufficient data
    if len(df) < 10:
        raise ValueError(f"Insufficient data after feature engineering for ticker '{ticker}'. Got {len(df)} rows.")
    
    predictors = ["Close", "Volume", "Open", "High", "Low"] + new_predictors
    return df, predictors
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# Trains a Random Forest model on the provided data and predictors.
# df: DataFrame with historical stock data and engineered features
# predictors: List of predictor column names
def train_model(df, predictors):
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1, class_weight='balanced')
    model.fit(df[predictors], df["Target"])
    return model
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# Predicts the probability and direction (Up/Down) for the next trading day.
# model: Trained RandomForestClassifier
# latest_data: DataFrame with the latest stock data (single row)
# predictors: List of predictor column names
def predict_next_day(model, latest_data, predictors):
    if latest_data.isnull().any().any():
        latest_data = latest_data.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
    prob_up = model.predict_proba(latest_data[predictors])[:, 1][0]
    prediction = "Up" if prob_up >= 0.5 else "Down"
    return prediction, prob_up
# ---------------------------------------------------------------------------------