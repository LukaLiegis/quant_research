import yfinance as yf
from numba import jit
import pytz


def get_data(ticker):
    """
    Get data from yahoo finance
    :param ticker:
    :return: dataframe
    """
    df = yf.Ticker(ticker).history(period="50y", auto_adjust=True).reset_index()
    df = df.rename(columns={
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    df["datetime"] = df["datetime"].dt.tz_convert(pytz.utc)
    df["dom"] = df["datetime"].dt.day
    # Calculate the percent change os the close data
    df["returns"] = df["close"].pct_change()
    df = df.drop(columns=["Dividends", "Stock Splits", "Capital Gains"])
    df = df.set_index("datetime", drop=True)
    df = df.dropna()
    print("The size of ", ticker, "is ", df.shape)
    return df
