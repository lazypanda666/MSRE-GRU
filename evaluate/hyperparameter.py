#!/usr/bin/env python
# coding=utf-8
"""
@Description   超参搜索评估
@Author        Alex_McAvoy
@Date          2026-04-14 16:20:24
"""

import os
import pickle
import numpy as np
import sklearn.metrics as sm

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes

def _saveEvaluateResult(report: dict, filename: str, save_path: str) -> None:
    """
    @description 保存评估结果
    @param {dict} report 评估结果字典
    @param {str} filename 存储文件名前缀
    @param {str} save_path 存储路径
    @return {None}
    """
    # 存储报告
    with open(save_path + f"{filename}.txt", "w", encoding="UTF-8") as f:
        f.write(f"{filename} 评估结果\n")
        f.write("--------------------------------------------\n")

        f.write(f"准确率：{report['accuracy']}\n")
        f.write(f"正确分类的样本数：{report['accuracy_num']}\n")
        f.write(f"微平均准确率：{report['micro_precision']}\n")
        f.write(f"微平均召回率：{report['micro_recall']}\n")
        f.write(f"微平均F1得分：{report['micro_f1']}\n")
        f.write(f"宏分类报告：\n{report['macro_class_report']}\n")

@calTimes(logger, "评估完成")
def evaluateHyperparameterModel(input_dir: str = cfg.MODEL["file_path"],
                                sampled: str = "sampled",
                                attack_type: dict = cfg.DATASET["attack_type"],
                                decimal_places: int = cfg.EVALUATE["decimal_places"],
                                output_dir: str = cfg.EVALUATE["file_path"]) -> None:
    """
    @description 评估超参搜索下的各模型
    @param {str} sampled 数据获取方式，sampled采样，unsampled未采样
    @param {dict} attack_type 攻击行为映射字典
    @param {int} decimal_places 结果小数位数
    @param {str} output_dir 存储路径
    @return {None}
    """
    bsu_dir = input_dir + f"{cfg.MODEL['name']}/BSU/"
    stu_dir = input_dir + f"{cfg.MODEL['name']}/STU/"
    model_dir = input_dir + f"{cfg.MODEL['name']}/Classifier/"
    save_path = output_dir + f"{cfg.MODEL['name']}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for kappa in [0.0, 0.5, 1.0, 1.5, 2.0]:
        bsu_data = np.load(bsu_dir + f"select_val_{sampled}_{kappa}.npy")
        bsu_X = bsu_data[:, :-1].astype(np.float32)
        for k in [200, 250, 300, 350, 400, 450, 500]:
            if bsu_X.shape[1] <= k:
                continue
            for rho in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                print("---------------------")
                print(f"正在评估：kappa={kappa}, k={k}, rho={rho}")
                # 加载数据与模型
                val_data = np.load(stu_dir + f"spectrum_val_{sampled}_{kappa}_{k}_{rho}.npy")
                X_val = val_data[:, :-1].astype(np.float32)
                y_val = val_data[:, -1].astype(np.uint8)
                with open(model_dir + f"classifier_{sampled}_{kappa}_{k}_{rho}.pkl", "rb") as f:
                    model = pickle.load(f)

                # 将攻击行为映射中文标签
                attack_type = {int(k): v for k, v in attack_type.items()}
                unique_labels = sorted(np.unique(y_val))
                attack_labels = [attack_type[label] for label in unique_labels if label in attack_type]

                # 预测结果
                y_pred = model.predict(X_val)

                # 准确率
                accuracy = round(sm.accuracy_score(y_val, y_pred), decimal_places)
                # 正确分类的样本数
                accuracy_num = sm.accuracy_score(y_val, y_pred, normalize=False)
                # macro 分类报告
                macro_class_report = sm.classification_report(y_val, y_pred, target_names=attack_labels, digits=decimal_places)
                # 微精确率
                micro_precision = round(sm.precision_score(y_val, y_pred, average="micro"), decimal_places)
                # 微召回率
                micro_recall = round(sm.recall_score(y_val, y_pred, average="micro"), decimal_places)
                # 微F1得分
                micro_f1 = round(sm.f1_score(y_val, y_pred, average="micro"), decimal_places)
                # 混淆矩阵
                cm = sm.confusion_matrix(y_val, y_pred)

                # 生成报告字典
                report = {
                    "labels": attack_labels,
                    "accuracy": accuracy,
                    "accuracy_num": accuracy_num,
                    "micro_precision": micro_precision,
                    "micro_recall": micro_recall,
                    "micro_f1": micro_f1,
                    "macro_class_report": macro_class_report,
                    "confusion_matrix": cm
                }

                _saveEvaluateResult(report, f"{sampled}_{kappa}_{k}_{rho}", save_path)

                print(f"准确率：{report['accuracy']}")
                print(f"正确分类的样本数：{report['accuracy_num']}")
                print(f"微平均准确率：{report['micro_precision']}")
                print(f"微平均召回率：{report['micro_recall']}")
                print(f"微平均F1得分：{report['micro_f1']}")
                print(f"宏分类报告：\n{report['macro_class_report']}")

