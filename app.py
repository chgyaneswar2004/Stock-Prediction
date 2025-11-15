import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Load the model
try:
    model = load_model('stock_dl_model.h5')
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock', 'POWERGRID.NS').strip()
        if not stock:
            return render_template('index.html', error="Stock ticker cannot be empty.")

        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()

        try:
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                raise ValueError("No data found for the given stock ticker.")
        except Exception as e:
            return render_template('index.html', error=f"Error fetching data: {e}")

        # Data processing and plotting
        try:
            data_desc = df.describe()

            # EMA calculations
            for span in [20, 50, 100, 200]:
                df[f'EMA{span}'] = df['Close'].ewm(span=span, adjust=False).mean()

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], 'y', label='Closing Price')
            plt.plot(df['EMA20'], 'g', label='EMA 20')
            plt.plot(df['EMA50'], 'r', label='EMA 50')
            plt.title("Closing Price vs Time (20 & 50 Days EMA)")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            ema_chart_path = "static/ema_20_50.png"
            plt.savefig(ema_chart_path)
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], 'y', label='Closing Price')
            plt.plot(df['EMA100'], 'g', label='EMA 100')
            plt.plot(df['EMA200'], 'r', label='EMA 200')
            plt.title("Closing Price vs Time (100 & 200 Days EMA)")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            ema_chart_path_100_200 = "static/ema_100_200.png"
            plt.savefig(ema_chart_path_100_200)
            plt.close()

            # Data splitting and scaling
            data_training = df[['Close']][:int(len(df) * 0.70)]
            data_testing = df[['Close']][int(len(df) * 0.70):]

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Prepare data for prediction
            if model:
                past_100_days = data_training.tail(100)
                final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
                input_data = scaler.transform(final_df)

                x_test, y_test = [], []
                for i in range(100, input_data.shape[0]):
                    x_test.append(input_data[i - 100:i])
                    y_test.append(input_data[i, 0])

                x_test, y_test = np.array(x_test), np.array(y_test)
                y_predicted = model.predict(x_test)

                scale_factor = 1 / scaler.scale_[0]
                y_predicted = y_predicted * scale_factor
                y_test = y_test * scale_factor

                plt.figure(figsize=(12, 6))
                plt.plot(y_test, 'g', label="Original Price")
                plt.plot(y_predicted, 'r', label="Predicted Price")
                plt.title("Prediction vs Original Trend")
                plt.xlabel("Time")
                plt.ylabel("Price")
                plt.legend()
                prediction_chart_path = "static/stock_prediction.png"
                plt.savefig(prediction_chart_path)
                plt.close()
            else:
                prediction_chart_path = None

            # Save dataset as CSV
            csv_file_path = f"static/{stock}_dataset.csv"
            df.to_csv(csv_file_path)

            return render_template('index.html',
                                   plot_path_ema_20_50=ema_chart_path,
                                   plot_path_ema_100_200=ema_chart_path_100_200,
                                   plot_path_prediction=prediction_chart_path,
                                   data_desc=data_desc.to_html(classes='table table-bordered'),
                                   dataset_link=csv_file_path)
        except Exception as e:
            return render_template('index.html', error=f"An error occurred during data processing: {e}")

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
