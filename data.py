import yfinance as yf
from numba import jit
import pytz


def get_data(ticker, period):
    """
    Get data from yahoo finance
    :param period: how much data to get
    :param ticker:
    :return: dataframe
    """
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True).reset_index()
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
    df = df.drop(columns=["Dividends", "Stock Splits", "Capital Gains"], errors="ignore")
    df = df.set_index("datetime", drop=True)
    df = df.dropna()
    print("The size of ", ticker, "is ", df.shape)
    return df
