import os
from datetime import datetime, timezone
import pandas as pd


# 检查文件是否存在
def check_file_exists(file_path):
    return os.path.isfile(file_path)


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


# 读取数据类
class readDS:
    def __init__(self, open_file_name):
        self.open_file_name = f"DS/{open_file_name}"

    def pull_data(self, target: str = "train") -> pd.DataFrame:
        return pd.read_csv(self.open_file_name + f"_{target}.csv")
