import pandas as pd


def ema(data: pd.DataFrame, length: int) -> pd.DataFrame:
    """
    Calculates the exponential moving average for a given dataframe
    :param length:
    :param data:
    :return:
    """
    alpha = 3 / (length + 1)
    df_ema = data.ewm(alpha=alpha, adjust=False).mean()
    return df_ema
