# Stock-Sense: Stock Trend Prediction

Stock-Sense is a web application that predicts stock trends using a deep learning model. It provides visualizations of historical stock data, exponential moving averages (EMAs), and predicted stock prices.

## Features

- **Stock Data Visualization**: View historical closing prices, EMAs (20, 50, 100, and 200 days), and predicted stock trends.
- **Customizable Tickers**: Enter any valid stock ticker to get predictions.
- **Data Export**: Download the historical dataset as a CSV file.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/stock_price_prediction.git
   cd stock_price_prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   gunicorn app:app
   ```

5. **Access the application:**
   Open your web browser and go to `http://127.0.0.1:8000`.

## Usage

- **Enter a stock ticker** in the input field and click "Submit."
- **View the generated charts** for EMA trends and price predictions.
- **Download the dataset** by clicking the "Download Dataset (CSV)" button.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `flask`
- `yfinance`
- `scikit-learn`
- `gunicorn`
