import pandas as pd
import os
from datetime import datetime, timezone


# 工具类
class Tools:
    # 检查文件是否存在
    @staticmethod
    def check_file_exists(file_path):
        return os.path.isfile(file_path)

    # 将毫秒级时间戳转换为秒
    @staticmethod
    def timestampToLocalTime(timestamp):
        timestamp_seconds = int(timestamp) / 1000
        # 使用fromtimestamp方法将时间戳转换为datetime对象
        utc_time = datetime.fromtimestamp(timestamp_seconds, timezone.utc)
        # 使用astimezone方法将UTC时间转换为本地时间
        local_time = utc_time.astimezone()
        # 使用strftime方法将时间格式化为字符串
        formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_time


# 指标计算类
class Indicators:
    # 计算SMA
    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int = 30, key: str = "close") -> None:
        df[f"SMA_{key[0]}{period}"] = df[key].rolling(window=period).mean()

    # 计算macd
    @staticmethod
    def calculate_macd(
        df: pd.DataFrame, period1: int = 12, period2: int = 26, period3: int = 9
    ) -> None:
        EMA_a = df["close"].ewm(span=period1, adjust=False).mean()
        EMA_b = df["close"].ewm(span=period2, adjust=False).mean()
        df["MACD"] = EMA_a - EMA_b
        df["MACD信号线"] = df["MACD"].ewm(span=period3, adjust=False).mean()

    # 计算rsi
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> None:
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

    # 计算布林带
    @staticmethod
    def calculate_band(df: pd.DataFrame, period: int = 20) -> None:
        band_Middle = df["close"].rolling(window=period).mean()
        df["band-Upper"] = band_Middle + 2 * df["close"].rolling(window=period).std()
        df["band-Lower"] = band_Middle - 2 * df["close"].rolling(window=period).std()

    # 计算成交量加权平均价格 (VWAP)
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> None:
        Typical_Price = (df["close"] + df["high"] + df["low"]) / 3
        df["VWAP"] = (Typical_Price * df["volume"]).cumsum() / df["volume"].cumsum()

    # 计算平均真实波幅 (ATR)
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> None:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = high_low.combine(high_close, max).combine(low_close, max)
        df["ATR"] = tr.rolling(window=period).mean()

    # 计算随机振荡器 (Stochastic Oscillator)
    @staticmethod
    def calculate_so(df: pd.DataFrame, period1: int = 14, period2: int = 3) -> None:
        Lowest_Low = df["low"].rolling(window=period1).min()
        Highest_High = df["high"].rolling(window=period1).max()
        df["%K"] = 100 * (df["close"] - Lowest_Low) / (Highest_High - Lowest_Low)
        df["%D"] = df["%K"].rolling(window=period2).mean()

    # 计算量比(Volume Ratio)
    @staticmethod
    def calculate_av(df: pd.DataFrame, period: int = 5) -> None:
        # 计算过去n天的平均成交量
        Average_Volume = df["volume"].rolling(window=period).mean()
        df[f"volRatio_{period}"] = df["volume"] / Average_Volume

    # 计算量价比(Volume-Price Ratio, VPR)
    @staticmethod
    def calculate_vpr(df: pd.DataFrame) -> None:
        df["VPR"] = df["volume"] / df["close"]

    # 数据归一化
    @staticmethod
    def normalization(df: pd.DataFrame) -> None:
        # 需要依照close归一化的列
        needed_close = ["band-Upper", "band-Lower"]
        # 需要进行变化量归一化的列
        needed_change = ["VWAP"]
        # 需要进行跌涨幅归一化的列
        needed_change_ad = ["VPR"]
        df["涨幅"] = ((df["close"] - df["close"].shift(1)) / df["close"].shift(1)) * 100
        df["振幅"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
        df["量涨幅"] = (df["volume"] - df["volume"].shift(1)) / df["volume"].shift(1)
        for col_name in df.columns:
            if col_name in needed_close:
                print(f"依照close归一化{col_name}")
                df[col_name] = (df[col_name] - df["close"]) / df["close"] * 100
            elif col_name in needed_change:
                print(f"依照变化量归一化{col_name}")
                df[col_name] = df[col_name] - df[col_name].shift(1)
            elif col_name in needed_change_ad:
                print(f"依照跌涨幅归一化{col_name}")
                df[col_name] = (
                    (df[col_name] - df[col_name].shift(1)) / df[col_name].shift(1)
                ) * 100
            elif "SMA_c" in col_name:
                df[col_name] = (df[col_name] - df["close"]) / df["close"] * 100
            elif "SMA_v" in col_name:
                df[col_name] = (df[col_name] - df["volume"]) / df["volume"]


# 数据处理类
class DataTreating:
    def __init__(self, open_file_name="DS/BTC-USDT-SWAP-15m") -> None:
        self.open_file_name = open_file_name
        self.testDataLength = 0

    def treatingData(self) -> None:
        if not Tools.check_file_exists(self.open_file_name + ".csv"):
            exit("csv文件不存在")
        df = pd.read_csv(self.open_file_name + ".csv")
        rows_to_drop = df[df["volume"] == 0.0]
        df = df[df["volume"] != 0.0]
        print(f"删除{rows_to_drop.shape[0]}行\t{rows_to_drop}")
        print(f"保留{df.shape[0]}行")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms").dt.strftime("%Y-%m-%d %H:%M")
        Indicators.calculate_macd(df)
        Indicators.calculate_rsi(df)
        Indicators.calculate_band(df)
        Indicators.calculate_vwap(df)
        Indicators.calculate_atr(df)
        Indicators.calculate_so(df)
        Indicators.calculate_av(df)
        Indicators.calculate_vpr(df)
        Indicators.calculate_sma(df, 4, "close")
        Indicators.calculate_sma(df, 16, "close")
        Indicators.calculate_sma(df, 48, "close")
        Indicators.calculate_sma(df, 96, "close")
        Indicators.calculate_sma(df, 288, "close")
        Indicators.calculate_sma(df, 4, "volume")
        Indicators.calculate_sma(df, 16, "volume")
        Indicators.calculate_sma(df, 48, "volume")
        Indicators.calculate_sma(df, 96, "volume")
        Indicators.calculate_sma(df, 288, "volume")
        Indicators.normalization(df)
        del df["open"]
        del df["high"]
        del df["low"]
        del df["volCcy"]
        del df["volCcyQuote"]
        del df["confirm"]
        self.testDataLength = int(df.shape[0] * 0.95)
        print(
            f"训练集长度{self.testDataLength}\t测试集长度{df.shape[0] - self.testDataLength}"
        )
        df[300 : self.testDataLength].to_csv(
            self.open_file_name + "_train.csv", index=False
        )
        df[self.testDataLength :].to_csv(self.open_file_name + "_test.csv", index=False)

    @staticmethod
    def check_data_is_NaN(df: pd.DataFrame) -> bool:
        # 检查每列是否有 NaN 值
        nan_columns = df.isna().any()
        print("Columns with NaN values:")
        print(nan_columns)
        # 获取包含 NaN 值的行
        rows_with_nan = df[df.isna().any(axis=1)]
        print("Rows with NaN values:")
        print(rows_with_nan)
        print(df.shape)
        df_cleaned = df.dropna(axis=0)
        print(df_cleaned.shape, df_cleaned.shape[0] - df.shape[0])
        return True


# 读取数据类
class readDS:
    def __init__(self, open_file_name):
        self.open_file_name = open_file_name

    def pull_data(self, target="train") -> pd.DataFrame:
        return pd.read_csv(self.open_file_name + f"_{target}.csv")


if __name__ == "__main__":
    # dataTreating = DataTreating("DS/BTC-USDT-SWAP-1D")
    # dataTreating.treatingData()
    readDSObject = readDS("DS/BTC-USDT-SWAP-1D")
    res = readDSObject.pull_data()
    print(res.columns)
    DataTreating.check_data_is_NaN(res)
