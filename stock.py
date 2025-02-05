import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
from textblob import TextBlob

# Set up the app layout
st.set_page_config(page_title="Stock Analysis", layout="wide")
st.title("📊 Stock Analysis Dashboard")
st.sidebar.header("⚙ Settings")

# Sidebar inputs for ticker symbol and date range
ticker = st.sidebar.text_input("Enter Ticker Symbol", "", help="Enter the stock ticker symbol, e.g., AAPL for Apple.")
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1), help="Select the start date for analysis.")
end_date = st.sidebar.date_input("End Date", datetime(2025, 1, 1), help="Select the end date for analysis.")

# Ensure start_date is earlier than end_date
if start_date >= end_date:
    st.sidebar.error("Start Date must be earlier than End Date.")

# Function to analyze sentiment of news headlines
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Function to fetch sample stock-related news (static for demo)
def fetch_sample_news(ticker):
    return [
        f"{ticker} reports impressive quarterly earnings, stock rises.",
        f"Stock prices for {ticker} drop after disappointing earnings report.",
        f"New product launch by {ticker} boosts stock price significantly.",
        f"{ticker} faces regulatory challenges, stock declines."
    ]

# Sentiment analysis for the headlines
def sentiment_analysis_on_news(news_list):
    results = []
    for news in news_list:
        polarity, subjectivity = analyze_sentiment(news)
        results.append({
            "headline": news,
            "polarity": polarity,
            "subjectivity": subjectivity
        })
    return pd.DataFrame(results)

if ticker:
    st.write("### Sentiment Analysis of Stock-related Headlines")
    sample_news = fetch_sample_news(ticker)
    sentiment_df = sentiment_analysis_on_news(sample_news)
    st.dataframe(sentiment_df)

    # Overall Sentiment
    average_polarity = sentiment_df["polarity"].mean()
    if average_polarity > 0:
        sentiment_status = "Positive"
    elif average_polarity < 0:
        sentiment_status = "Negative"
    else:
        sentiment_status = "Neutral"
    st.write(f"### Overall Sentiment: {sentiment_status} (Polarity: {average_polarity:.2f})")

    average_subjectivity = sentiment_df["subjectivity"].mean()
    st.write(f"### Average Subjectivity: {average_subjectivity:.2f} (Higher values indicate more opinion-based news)")

# Load fundamental data
@st.cache_data
def load_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {
            "Company Name": info.get("longName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Country": info.get("country"),
            "Revenue (TTM)": info.get("totalRevenue"),
            "Gross Profit (TTM)": info.get("grossProfits"),
            "EBITDA": info.get("ebitda"),
            "Net Income (TTM)": info.get("netIncomeToCommon"),
            "P/E Ratio (TTM)": info.get("trailingPE"),
            "Forward P/E Ratio": info.get("forwardPE"),
            "Price-to-Sales Ratio": info.get("priceToSalesTrailing12Months"),
        }
        return fundamentals
    except Exception as e:
        st.error(f"Error loading fundamentals: {e}")
        return None

# Display fundamentals
if ticker:
    fundamentals = load_fundamentals(ticker)
    if fundamentals:
        st.write("### 📋 Fundamentals")
        st.table(pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"]))

# Load stock price data
@st.cache_data
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            st.error("No data available for this ticker in the selected date range.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

if ticker and start_date and end_date:
    df = load_data(ticker, start_date, end_date)

    if df is not None and not df.empty:
        st.write(f"### Stock Data for {ticker}")
        st.write(f"Data from {start_date} to {end_date}: ")
        st.dataframe(df.head(), use_container_width=True)

        # Extract Close prices for analysis
        data = df[['Close']].copy()

        # Calculate Moving Averages
        def calculate_moving_averages(data, short_window=50, long_window=200):
            data['SMA_50'] = data['Close'].rolling(window=short_window).mean()
            data['SMA_200'] = data['Close'].rolling(window=long_window).mean()
            return data

        data = calculate_moving_averages(data)

        # Plot Moving Averages
        st.write(f"### 📈 Moving Averages for {ticker}")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.6)
        ax.plot(data.index, data['SMA_50'], label='50-Day SMA', color='orange', alpha=0.8)
        ax.plot(data.index, data['SMA_200'], label='200-Day SMA', color='green', alpha=0.8)
        ax.set_title(f'{ticker} Moving Averages', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend()
        st.pyplot(fig)

        # Bullish or Bearish Trend
        st.write("### Bullish or Bearish Insights")
        if data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1]:
            st.success("🔔 The stock is in a bullish trend (Golden Cross detected).")
        elif data['SMA_50'].iloc[-1] < data['SMA_200'].iloc[-1]:
            st.warning("⚠ The stock is in a bearish trend (Death Cross detected).")
        else:
            st.info("The moving averages are converging. Watch for potential crossovers.")

        # RSI Calculation
        def calculate_rsi(data, window=14):
            delta = data['Close'].diff(1)
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            avg_gain = gains.rolling(window=window).mean()
            avg_loss = losses.rolling(window=window).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            data['RSI'] = rsi
            return data

        data = calculate_rsi(data)

        # Plot RSI
        st.write(f"### RSI for {ticker}")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(data.index, data['RSI'], label='RSI', color='red')
        ax.axhline(70, color='green', linestyle='--', label='Overbought (70)')
        ax.axhline(30, color='orange', linestyle='--', label='Oversold (30)')
        ax.set_title(f'{ticker} RSI (Relative Strength Index)', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('RSI Value', fontsize=12)
        ax.legend()
        st.pyplot(fig)

        # Stop-Loss Calculation
        def calculate_stop_loss(data, window=14):
            data['Stop_Loss'] = data['Close'].rolling(window=window).min()
            return data

        data = calculate_stop_loss(data)

        # Plot Stop-Loss Levels
        st.write(f"### Stop-Loss Levels for {ticker}")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(data.index, data['Close'], label='Close Price', color='blue')
        ax.plot(data.index, data['Stop_Loss'], label='Stop-Loss Level', color='orange', linestyle='--')
        ax.set_title(f'{ticker} Stop-Loss Levels', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("⚠ No data available. Please adjust the date range or check the ticker symbol.")
else:
    st.write("Please enter the ticker symbol, start date, and end date to begin the analysis.")