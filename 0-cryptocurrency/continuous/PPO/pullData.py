import pandas as pd
import numpy as np
import talib
import os
import pickle
from datetime import datetime, timezone

import config


def check_file_exists(file_path):
    return os.path.isfile(file_path)


def saveData(filename, save_ndarray):
    with open(filename, "wb") as fw:
        pickle.dump(save_ndarray, fw)
    return


def openData(filename):
    try:
        with open(filename, "rb") as fr:
            return_Mat = pickle.load(fr)
        return return_Mat
    except:
        return []


# 将毫秒级时间戳转换为秒
def timestampToLocalTime(timestamp):
    timestamp_seconds = int(timestamp) / 1000
    # 使用fromtimestamp方法将时间戳转换为datetime对象
    utc_time = datetime.fromtimestamp(timestamp_seconds, timezone.utc)
    # 使用astimezone方法将UTC时间转换为本地时间
    local_time = utc_time.astimezone()
    # 使用strftime方法将时间格式化为字符串
    formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


class PullData:

    def __init__(self, open_file_name="DS/BTC-USDT-SWAP-15m") -> None:
        self.open_file_name = open_file_name
        self.npyData: np.ndarray = np.array([])
        self.dfData: pd.DataFrame = self.ndarray_to_dataframe()
        self.testDataLength = int(self.dfData.shape[0] * 0.95)

    def calculate_sma(self, close, period=30):
        sma = talib.SMA(close, timeperiod=int(period))
        return sma

    def calculate_adx(self, high, low, close, period=14):
        adx = talib.ADX(high, low, close, timeperiod=period)
        return adx

    def calculate_macd(self, close) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        macd, macdsignal, macdhist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        return macd, macdsignal, macdhist

    def calculate_rsi(self, close):
        rsi = talib.RSI(close, timeperiod=14)
        return rsi

    def ndarray_to_dataframe(self) -> pd.DataFrame:
        if check_file_exists(self.open_file_name + ".csv"):
            print("csv文件存在,直接读取")
            return pd.read_csv(self.open_file_name + ".csv")
        self.npyData: np.ndarray = openData(self.open_file_name + ".npy")
        # print(self.npyData.shape)
        self.npyData = self.npyData[:, 0:6].astype(float)
        # 计算涨幅
        extent: np.ndarray = (self.npyData[:, 4] - self.npyData[:, 1]) / self.npyData[
            :, 1
        ]
        # 计算adx
        adx: np.ndarray = self.calculate_adx(
            self.npyData[:, 2], self.npyData[:, 3], self.npyData[:, 4]
        )
        # 计算rsi
        rsi: np.ndarray = self.calculate_rsi(self.npyData[:, 4])
        # 计算macd
        macd, macdsignal, macdhist = self.calculate_macd(self.npyData[:, 4])
        self.npyData = np.hstack(
            (
                self.npyData,
                extent.reshape(-1, 1),
                adx.reshape(-1, 1),
                rsi.reshape(-1, 1),
                macd.reshape(-1, 1),
                macdsignal.reshape(-1, 1),
                macdhist.reshape(-1, 1),
            )
        )
        df = pd.DataFrame(self.npyData)
        col_names = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "extent",
            "adx",
            "rsi",
            "macd",
            "macdsignal",
            "macdhist",
        ]
        df.columns = col_names
        df.to_csv(self.open_file_name + ".csv", index=False)
        return df

    def get_data(self, target="train") -> pd.DataFrame:
        # 检查列名列表中的列名是否全部有效
        valid_columns = set(self.dfData.columns)
        invalid_columns = [
            col for col in config.TECHNICAL_INDICATORS_LIST if col not in valid_columns
        ]
        if invalid_columns:
            print(f"以下列名无效: {invalid_columns}")

        for col_name in invalid_columns:
            [use, _, period] = col_name.split("_")
            if use == "c":
                ma = self.calculate_sma(self.dfData["close"], period)
                self.dfData[col_name] = ma
            elif use == "v":
                ma = self.calculate_sma(self.dfData["volume"], period)
                self.dfData[col_name] = ma
        # 将毫秒时间戳列转换为格式化日期时间字符串
        self.dfData["time"] = pd.to_datetime(
            self.dfData["time"], unit="ms"
        ).dt.strftime("%Y-%m-%d %H:%M")
        # 过滤DataFrame中的列
        filtered_df = self.dfData[config.TECHNICAL_INDICATORS_LIST]
        filtered_df[300 : self.testDataLength].to_csv(
            self.open_file_name + "_train.csv", index=False
        )
        filtered_df[self.testDataLength :].to_csv(
            self.open_file_name + "_test.csv", index=False
        )

        return (
            filtered_df[300 : self.testDataLength]
            if target == "train"
            else filtered_df[self.testDataLength :]
        )

    def pull_data(self, target="train") -> pd.DataFrame:
        if check_file_exists(self.open_file_name + f"_{target}.csv"):
            print("csv文件存在,直接读取")
            return pd.read_csv(self.open_file_name + f"_{target}.csv")


if __name__ == "__main__":
    pull_data = PullData()
    # print(pull_data.dfData[-5:])
    pull_data.get_data()
    print(pull_data.dfData[-5:])
    pass
