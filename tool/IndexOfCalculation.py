import pandas as pd


# 计算SMA
def calculate_sma(df: pd.DataFrame, period: int = 30, key: str = "close") -> None:
    df[f"SMA_{key[0]}{period}"] = df[key].rolling(window=period).mean()


# 计算macd
def calculate_macd(
    df: pd.DataFrame, period1: int = 12, period2: int = 26, period3: int = 9
) -> None:
    # 计算EMA
    ema1 = df["close"].ewm(span=period1, adjust=False).mean()
    ema2 = df["close"].ewm(span=period2, adjust=False).mean()
    # 计算DIF
    df["dif"] = ema1 - ema2
    # 计算DEA
    df["dea"] = df["dif"].ewm(span=period3, adjust=False).mean()
    # 计算MACD
    df["macd"] = 2 * (df["dif"] - df["dea"])


# 计算rsi
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> None:
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))


# 计算布林带
def calculate_band(df: pd.DataFrame, period: int = 20) -> None:
    band_Middle = df["close"].rolling(window=period).mean()
    df["band-Upper"] = band_Middle + 2 * df["close"].rolling(window=period).std()
    df["band-Lower"] = band_Middle - 2 * df["close"].rolling(window=period).std()


# 计算成交量加权平均价格 (VWAP)
def calculate_vwap(df: pd.DataFrame) -> None:
    Typical_Price = (df["close"] + df["high"] + df["low"]) / 3
    df["VWAP"] = (Typical_Price * df["volume"]).cumsum() / df["volume"].cumsum()


# 计算平均真实波幅 (ATR)
def calculate_atr(df: pd.DataFrame, period: int = 14) -> None:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df["ATR"] = tr.rolling(window=period).mean()


# 计算随机振荡器 (Stochastic Oscillator)
def calculate_so(df: pd.DataFrame, period1: int = 14, period2: int = 3) -> None:
    Lowest_Low = df["low"].rolling(window=period1).min()
    Highest_High = df["high"].rolling(window=period1).max()
    df["%K"] = 100 * (df["close"] - Lowest_Low) / (Highest_High - Lowest_Low)
    df["%D"] = df["%K"].rolling(window=period2).mean()


# 计算量比(Volume Ratio)
def calculate_av(df: pd.DataFrame, period: int = 5) -> None:
    # 计算过去n天的平均成交量
    Average_Volume = df["volume"].rolling(window=period).mean()
    df[f"volRatio_{period}"] = df["volume"] / Average_Volume


# 计算量价比(Volume-Price Ratio, VPR)
def calculate_vpr(df: pd.DataFrame) -> None:
    df["VPR"] = df["volume"] / df["close"]
