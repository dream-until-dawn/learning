import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

TRAIN_LOG_DIR = "train_log/PPO"


def get_folders_in_directory(directory_path):
    # 获取指定目录下的所有文件和文件夹
    all_items = os.listdir(directory_path)

    # 过滤出文件夹
    folders = [
        item for item in all_items if os.path.isdir(os.path.join(directory_path, item))
    ]

    return folders


def merge_tensorboard_logs_by_time(log_dirs):
    print(log_dirs)
    # 创建一个SummaryWriter
    writer = SummaryWriter(f"{TRAIN_LOG_DIR}/merge")

    events = []
    tags_count_dict = {}

    # 读取每个日志目录中的事件文件
    for log_dir in log_dirs:
        if log_dir == "merge":
            continue
        read_log_dir = f"{TRAIN_LOG_DIR}/{log_dir}"
        event_acc = EventAccumulator(read_log_dir)
        event_acc.Reload()

        # 获取所有标量标签
        tags = event_acc.Tags()["scalars"]
        if tags[0] not in tags_count_dict:
            tags_count_dict = {tag: 0 for tag in tags}

        # 将每个标签的值写入新的日志文件
        for tag in tags:
            scalar_events = event_acc.Scalars(tag)
            for event in scalar_events:
                events.append((tag, event.wall_time, tags_count_dict[tag], event.value))
                tags_count_dict[tag] += 1
        print(tags_count_dict)

    # 写入新的日志文件
    for tag, wall_time, step, value in events:
        writer.add_scalar(tag, value, step, wall_time)

    writer.close()


def check_log():
    read_log_dir = f"{TRAIN_LOG_DIR}"
    event_acc = EventAccumulator(read_log_dir)
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]
    for tag in tags:
        scalar_events = event_acc.Scalars(tag)
        print(tag, len(scalar_events))
        print(tag, scalar_events[0])
        print(tag, scalar_events[-1])
        # for event in scalar_events:
        #     print(tag, event.step, event.value)


if __name__ == "__main__":
    log_dirs = get_folders_in_directory(TRAIN_LOG_DIR)
    merge_tensorboard_logs_by_time(log_dirs)

    # check_log()
