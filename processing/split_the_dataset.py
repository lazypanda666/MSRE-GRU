import os
import numpy as np
from sklearn.model_selection import train_test_split

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes

@calTimes(logger, "划分数据集完成")
def SplitTheDataset(input_dir: str = cfg.DISTRIBUTION["file_path"],
                    output_dir: str = cfg.DATA["file_path"]) -> None:


    X_list = []
    y_list = []

    # 遍历文件
    for file in os.listdir(input_dir):
        if file.startswith("sampled_") and file.endswith("_e.npy"):
            file_path = os.path.join(input_dir, file)
            
            # 读取数据
            data = np.load(file_path)   # shape: (N, d) 或 (N,)
            
            # 从文件名解析 label
            # sampled_{label}_e.npy
            label = file.split("_")[1]
            label = int(label)  # 如果是数字标签
            
            # 如果是一维数据，扩展维度
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # 构造标签列
            labels = np.full((data.shape[0], 1), label)
            
            X_list.append(data)
            y_list.append(labels)

    # 拼接所有数据
    X = np.vstack(X_list)
    y = np.vstack(y_list)

    # 合并特征和标签（最后一列为label）
    dataset = np.hstack([X, y])

    # 划分 7:3（建议分层抽样，保证类别均衡）
    train, test = train_test_split(
        dataset,
        test_size=0.3,
        random_state=36,
        stratify=dataset[:, -1]  # 按标签分层
    )

    # 保存
    train_path = os.path.join(output_dir, "train.npy")
    test_path = os.path.join(output_dir, "test.npy")

    np.save(train_path, train)
    np.save(test_path, test)

    logger.info(f"Train saved to: {train_path}, shape={train.shape}")
    logger.info(f"Test saved to: {test_path}, shape={test.shape}")