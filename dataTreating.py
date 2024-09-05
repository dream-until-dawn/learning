import pandas as pd
from tool.RWcorrelational import check_file_exists, readDS
from tool.IndexOfCalculation import calculate_macd


# 数据处理类
class DataTreating:
    def __init__(self, open_file_name="ETH-USDT-SWAP-1D") -> None:
        self.open_file_name = f"DS/{open_file_name}"
        self.testDataLength = 0

    def treatingData(self) -> None:
        if not check_file_exists(self.open_file_name + ".csv"):
            exit("csv文件不存在")
        df = pd.read_csv(self.open_file_name + ".csv")
        rows_to_drop = df[df["volume"] == 0.0]
        df = df[df["volume"] != 0.0]
        print(f"删除{rows_to_drop.shape[0]}行\t{rows_to_drop}")
        print(f"保留{df.shape[0]}行")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms").dt.strftime("%Y-%m-%d %H:%M")
        df = df.iloc[::-1]  # 反序
        calculate_macd(df)
        del df["volCcy"]
        del df["volCcyQuote"]
        del df["confirm"]
        self.testDataLength = int(df.shape[0] * 0.9)
        print(
            f"训练集长度{self.testDataLength}\t测试集长度{df.shape[0] - self.testDataLength}"
        )
        df[30 : self.testDataLength].to_csv(
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
        df_cleaned = df.dropna(axis=0)
        print(df_cleaned.shape, df_cleaned.shape[0] - df.shape[0])
        return True


if __name__ == "__main__":
    # 处理数据
    DataTreating("ETH-USDT-SWAP-1D").treatingData()
    pass
    # 检查数据
    trainData = readDS("ETH-USDT-SWAP-1D").pull_data()
    DataTreating.check_data_is_NaN(trainData)
    testData = readDS("ETH-USDT-SWAP-1D").pull_data("test")
    DataTreating.check_data_is_NaN(testData)
