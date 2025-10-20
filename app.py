from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
from datetime import date, timedelta, datetime
import os
import plotly.express as px # <-- Import Plotly

app = Flask(__name__)

# --- Section 1: API Keys and Clients ---
NEWSAPI_API_KEY = "76e4ed517fc6400990d1175f8086d047" 
newsapi = NewsApiClient(api_key=NEWSAPI_API_KEY)
vader_analyzer = SentimentIntensityAnalyzer()

# --- Section 2: Load Price Prediction Models ---
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "NFLX", "JPM", "BAC", "WMT", "KO", "PEP", "PFE", "INTC"] # Corrected GOOGL
gru_models = {}
scalers = {}
try:
    print("Loading all GRU price prediction models (using Keras 3)...")
    for ticker in tickers:
        model_path = f"models/gru_model_{ticker}.keras"
        scaler_path = f"scalers/scaler_{ticker}.gz"
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            gru_models[ticker] = tf.keras.models.load_model(model_path)
            scalers[ticker] = joblib.load(scaler_path)
        else:
            print(f"⚠️ Warning: Model or scaler file missing for {ticker}. Skipping...")
    print(f"✅ {len(gru_models)} GRU models and scalers loaded successfully.")
except Exception as e:
    print(f"❌ Error loading GRU models or scalers: {e}")

# --------------------------------------------------

@app.route('/')
def index():
    available_tickers = list(gru_models.keys())
    return render_template('index.html', tickers=available_tickers)

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    if ticker not in gru_models or ticker not in scalers:
        return "Error: Model for this ticker is not available."

    # --- Task 1: Get Stock Fundamentals ---
    fundamentals = {}
    previous_close = None
    try:
        stock_info = yf.Ticker(ticker).info
        if stock_info:
            fundamentals = stock_info
            previous_close = fundamentals.get('previousClose')
        else:
             fundamentals = {"Info": "No fundamental data available."}
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch fundamentals for {ticker}: {e}")
        fundamentals = {"Error": "Could not retrieve fundamental data."}

    # --- Task 2: Price Prediction & Get Data for Graph ---
    prediction_formatted = "N/A"
    prediction_value = None
    graph_html = None # Initialize graph HTML as None
    try:
        # Fetch data for the last year for the graph
        start_date_graph = (date.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date_graph = date.today().strftime('%Y-%m-%d')
        stock_data = yf.download(ticker, start=start_date_graph, end=end_date_graph)

        if stock_data.empty:
            print(f"⚠️ Warning: No data downloaded for {ticker}. Likely rate-limited by yfinance.")
            prediction_formatted = "Data unavailable"
        else:
            # --- Generate Plotly Graph ---
            try:
                fig = px.line(x=stock_data.index, y=stock_data['Close'].values.flatten(), title=f'{ticker} Stock Price (Last Year)')
                fig.update_layout(xaxis_title='Date', yaxis_title='Closing Price (USD)')
                # Convert figure to HTML (omitting full html tags, include only div)
                graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn') 
            except Exception as graph_e:
                print(f"❌ Error generating graph for {ticker}: {graph_e}")
                graph_html = "<p class='text-danger'>Could not generate graph.</p>"
            # ---------------------------

            # --- Prediction Logic (using the last 120 days of the fetched data) ---
            if ticker in gru_models and ticker in scalers:
                model = gru_models[ticker]
                scaler = scalers[ticker]
                N_STEPS = 60
                if len(stock_data) < N_STEPS:
                     print(f"⚠️ Warning: Not enough historical data ({len(stock_data)} days) for {ticker} to predict.")
                     prediction_formatted = "Not enough data"
                else:
                    last_60_days = stock_data['Close'].tail(N_STEPS).values.reshape(-1, 1)
                    if hasattr(scaler, 'scale_'): 
                        scaled_data = scaler.transform(last_60_days)
                        X_pred = np.reshape(scaled_data, (1, N_STEPS, 1))
                        prediction_scaled = model.predict(X_pred)
                        prediction_value = scaler.inverse_transform(prediction_scaled)[0][0]
                        prediction_formatted = f"{prediction_value:.2f}"
                    else:
                        print(f"❌ Error: Scaler for {ticker} was not properly fitted.")
                        prediction_formatted = "Scaler Error"
            else:
                 print(f"❌ KeyError: Model or scaler for '{ticker}' was not loaded for prediction.")
                 prediction_formatted = "Model not loaded"
            # --------------------------------------------------------------------

    except Exception as e:
        print(f"❌ Error during price prediction/data fetch for {ticker}: {e}")
        prediction_formatted = "Prediction failed"
        graph_html = "<p class='text-danger'>Could not fetch data for graph.</p>"


    # --- Task 3: Real-Time Sentiment Analysis (NewsAPI.org + VADER) ---
    sentiment_formatted = "N/A"
    sentiment_value = None
    try:
        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        all_articles = newsapi.get_everything(q=ticker,
                                              from_param=seven_days_ago,
                                              language='en',
                                              sort_by='relevancy', 
                                              page_size=20) 

        headlines = [article['title'] for article in all_articles.get('articles', []) if article.get('title')]
        
        if not headlines:
            sentiment_formatted = "No recent news via API"
        else:
            sentiment_scores = [vader_analyzer.polarity_scores(headline)['compound'] for headline in headlines]
            sentiment_value = np.mean(sentiment_scores) if sentiment_scores else 0 
            sentiment_formatted = f"{sentiment_value:.2f}" 
            
    except Exception as e:
        if hasattr(e, 'get_code'): 
            print(f"❌ NewsAPI Error ({e.get_code()}): {e.get_message()}")
            sentiment_formatted = f"API Error: {e.get_code()}"
        else:
            print(f"❌ Error fetching/analyzing news: {e}")
            sentiment_formatted = "News Error"

    # --- Task 4: Generate Trading Recommendation ---
    recommendation = "Hold" 
    recommendation_class = "text-secondary" 
    
    if prediction_value is not None and previous_close is not None and sentiment_value is not None:
        try:
            if prediction_value > (previous_close * 1.005) and sentiment_value > 0.05:
                recommendation = "Buy"
                recommendation_class = "text-success fw-bold" 
            elif prediction_value < (previous_close * 0.995) and sentiment_value < -0.05:
                recommendation = "Sell"
                recommendation_class = "text-danger fw-bold" 
            else:
                recommendation = "Hold"
                recommendation_class = "text-warning fw-bold" 
        except Exception as e:
            print(f"⚠️ Warning: Error calculating recommendation: {e}")
            recommendation = "Error"
            recommendation_class = "text-muted"
    else:
        recommendation = "N/A"
        recommendation_class = "text-muted"


    # --- Task 5: Display Results ---
    return render_template('result.html', 
                           ticker=ticker, 
                           prediction=prediction_formatted,
                           fundamentals=fundamentals, 
                           sentiment=sentiment_formatted,
                           recommendation=recommendation, 
                           recommendation_class=recommendation_class,
                           graph_html=graph_html) # <-- Pass graph HTML

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=False)