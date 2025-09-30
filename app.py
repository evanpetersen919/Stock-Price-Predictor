import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from model_utils import load_and_engineer_data, train_model, predict_next_day

# -------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="centered"
)

# -------------------- Title and Header ---------------------------
st.title("Stock Price Predictor")
st.write("Enter a stock ticker to view price history and get tomorrow's direction prediction.")

# -------------------- User Inputs -------------------------
st.subheader("Stock Information")

# Simple input layout
col1, col2 = st.columns(2)

with col1:
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Examples: AAPL, MSFT, TSLA").upper()

with col2:
    period_options = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}
    selected_period = st.selectbox("Time Period", options=list(period_options.keys()), index=3)
    period = period_options[selected_period]

# -------------------- Stock Data Display -------------------------
st.subheader(f"{ticker} Stock Price History")

# Fetch and display stock data
try:
    with st.spinner(f"Loading {ticker} data..."):
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
        else:
            # Create simple chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"{ticker} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display key metrics
            st.write("**Current Stock Information:**")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            current_price = df['Close'].iloc[-1]
            period_high = df['High'].max()
            period_low = df['Low'].min()
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
            change_percent = (price_change / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
            
            with metrics_col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({change_percent:+.1f}%)")
            
            with metrics_col2:
                st.metric("Period High", f"${period_high:.2f}")
            
            with metrics_col3:
                st.metric("Period Low", f"${period_low:.2f}")
            
except Exception as e:
    st.error(f"Error loading data: {str(e)}")

# -------------------- Prediction Section -------------------------
st.subheader("Tomorrow's Price Prediction")

if st.button("Get Prediction", type="primary"):
    try:
        with st.spinner("Training model and making prediction..."):
            # Load data and train model
            df_ml, predictors = load_and_engineer_data(ticker, period="max")
            
            if len(df_ml) < 2:
                st.error("Not enough data for prediction.")
            else:
                # Split data
                train = df_ml.iloc[:-1]
                test = df_ml.iloc[[-1]]
                
                # Train model and predict
                model = train_model(train, predictors)
                prediction, prob_up = predict_next_day(model, test, predictors)
                
                # Display prediction results
                st.success("Prediction Complete!")
                
                # Simple prediction display
                direction_color = "green" if prediction == "Up" else "red"
                confidence = max(prob_up, 1 - prob_up) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Prediction:** :{direction_color}[{prediction}]")
                with col2:
                    st.markdown(f"**Confidence:** {confidence:.1f}%")
                
                # Simple probability display
                st.write(f"Probability Up: {prob_up*100:.1f}% | Probability Down: {(1-prob_up)*100:.1f}%")
                
                # Warning disclaimer
                st.warning("**Disclaimer:** This prediction is for educational purposes only. Past performance doesn't guarantee future results. Always do your own research before making investment decisions.")
                
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Try using a different ticker or check your internet connection.")

# -------------------- About Section -------------------------
st.markdown("---")
st.markdown("**About:** This app uses machine learning to predict stock price direction. Data from Yahoo Finance. For educational purposes only.")
st.markdown("**Popular Tickers:** AAPL, MSFT, GOOGL, TSLA, NVDA, ^GSPC (S&P 500)")
