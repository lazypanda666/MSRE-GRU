#!/usr/bin/env python
# coding=utf-8
"""
@Description   模型评估
@Author        Alex_McAvoy
@Date          2025-09-21 23:24:51
"""
import os
import torch
import numpy as np
import sklearn.metrics as sm
import sklearn.preprocessing as sp
from evaluate.draw import plotConfusionMatrix
from model.GRU import ECGRU, load_entropy_sequence
from const import cfg
from utils.log import logger
from utils.wrapper import calTimes

def _saveEvaluateResult(filename_suffix: str, res: dict, save_path: str) -> None:
    """
    @description 保存评估结果
    @param {str} filename_suffix 存储文件名前缀
    @param {dict} res 评估结果字典
    @param {str} save_path 存储路径
    @return {None}
    """
    # 存储报告
    with open(save_path + f"{filename_suffix}_report.txt", "w", encoding="UTF-8") as f:
        f.write(f"{filename_suffix} 评估结果\n")
        f.write("--------------------------------------------\n")

        f.write(f"准确率：{res['accuracy']}\n")
        f.write(f"正确分类的样本数：{res['accuracy_num']}\n")
        f.write(f"微平均准确率：{res['micro_precision']}\n")
        f.write(f"微平均召回率：{res['micro_recall']}\n")
        f.write(f"微平均F1得分：{res['micro_f1']}\n")
        f.write(f"宏分类报告：\n{res['macro_class_report']}\n")

def computeMacroCurves(y_test: np.ndarray, y_score: np.ndarray) -> dict:
    """
    @description 针对多分类任务计算macro PR曲线与macro ROC曲线
    @param {np.ndarray} y_test 测试集真实标签
    @param {np.ndarray} y_score 模型输出得分
    @return {dict} 包含macro PR与macro ROC曲线数据的字典
    """

    # 获取所有类别
    unique_labels = np.unique(y_test)

    # 真实标签二值化 One-vs-Rest
    y_test_bin = sp.label_binarize(y_test, classes=unique_labels)

    # 获取类别数量
    num_classes = y_test_bin.shape[1]

    # 各类别PR曲线列表
    precision_ls = []
    recall_ls = []
    # 各类别ROC曲线列表
    fpr_ls = []
    tpr_ls = []
    # ======== 对每个类别分别计算 PR/ROC =========
    for i in range(num_classes):
        # 单类别PR
        p, r, _ = sm.precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        precision_ls.append(p)
        recall_ls.append(r)

        # 单类别ROC
        fpr, tpr, _ = sm.roc_curve(y_test_bin[:, i], y_score[:, i])
        fpr_ls.append(fpr)
        tpr_ls.append(tpr)

    # ======== 计算 macro PR ========
    # 生成统一recall轴
    recall_common = np.linspace(0, 1, 200)
    # 用于累加插值得到macro值
    precision_interpolated = np.zeros_like(recall_common)

    # 遍历每个类别的PR曲线
    for p, r in zip(precision_ls, recall_ls):
        # 对recall进行插值，使得不同类别的PR可以求平均
        precision_interpolated += np.interp(recall_common, r[::-1], p[::-1])

    # macro PR曲线
    precision_macro = precision_interpolated / num_classes
    # macro PR-AUC曲线
    pr_auc_macro = sm.auc(recall_common, precision_macro)

    # ======== 计算 macro ROC ========
    # 生成统一FPR轴
    fpr_common = np.linspace(0, 1, 200)
    # 用于累加插值得到macro值
    tpr_interpolated = np.zeros_like(fpr_common)

    # 遍历每个类别ROC曲线
    for fpr, tpr in zip(fpr_ls, tpr_ls):
        # 对fpr进行插值，使得不同类别的FPR可以求平均
        tpr_interpolated += np.interp(fpr_common, fpr, tpr)

    # macro ROC曲线
    tpr_macro = tpr_interpolated / num_classes
    # macro ROC-AUC
    roc_auc_macro = sm.auc(fpr_common, tpr_macro)

    # 返回所有数据
    return {
        "precision_macro": precision_macro,
        "recall_common": recall_common,
        "pr_auc_macro": pr_auc_macro,
        "fpr_macro": fpr_common,
        "tpr_macro": tpr_macro,
        "roc_auc_macro": roc_auc_macro
    }

@calTimes(logger, "评估完成")
def evaluateModel(model: object = cfg.MODEL["file_path"],
                  test_data: np.ndarray = cfg.DATA["file_path"] + "test.npy",
                  attack_type: dict = cfg.DATASET["attack_type"],
                  decimal_places: int = cfg.EVALUATE["decimal_places"],
                  cm_flag: bool = True,
                  save_path: str = cfg.EVALUATE["file_path"]) -> None:
    """
    @description 评估模型
    @param {object} model 模型
    @param {np.ndarray} test_data 测试数据
    @param {dict} attack_type 攻击行为映射字典
    @param {int} decimal_places 结果小数位数
    @param {bool} cm_flag 是否绘制混淆矩阵
    @param {str} save_path 存储路径
    @return {None}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_path = os.path.join(cfg.DATA["file_path"], "test.npy")
    X, delta_e, y_true = load_entropy_sequence(
        test_path,
        cfg.MODEL["seq_len"]
    )

    X = torch.tensor(X, dtype=torch.float32).to(device)
    delta_e = torch.tensor(delta_e, dtype=torch.float32).to(device)
    y_true = torch.tensor(y_true, dtype=torch.long).to(device)

    # 加载模型
    model_path = os.path.join(cfg.MODEL["file_path"], "ecgru.pth")
    model = ECGRU(
        input_dim=X.shape[-1],
        hidden_dim=cfg.MODEL["hidden_dim"],
        entropy_dim=X.shape[-1],
        num_classes=len(np.unique(y_true))
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    

    # 将攻击行为映射字典的键转为整数，方便匹配
    attack_type = {int(k): v for k, v in attack_type.items()}
    # 获取y_test中的唯一编号
    unique_labels = sorted(np.unique(y_true))
    # 映射成中文标签
    attack_labels = [attack_type[label] for label in unique_labels if label in attack_type]

    # ============= 模型预测 =============
    # 获取模型预测结果
    with torch.no_grad():
        logits = model(X, X, delta_e)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()

    # ============= 分类报告 =============
    # 准确率
    accuracy = round(sm.accuracy_score(y_true, y_pred), decimal_places)
    # 正确分类的样本数
    accuracy_num = sm.accuracy_score(y_true, y_pred, normalize=False)
    # macro 分类报告
    macro_class_report = sm.classification_report(y_true, y_pred, target_names=attack_labels, digits=decimal_places)
    # 微精确率
    micro_precision = round(sm.precision_score(y_true, y_pred, average="micro"), decimal_places)
    # 微召回率
    micro_recall = round(sm.recall_score(y_true, y_pred, average="micro"), decimal_places)
    # 微F1得分
    micro_f1 = round(sm.f1_score(y_true, y_pred, average="micro"), decimal_places)
    # 混淆矩阵
    cm = sm.confusion_matrix(y_true, y_pred)

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

    
    _saveEvaluateResult(f"MSRE-GRU", report, save_path)

    print(f"MSRE-GRU 模型评估结果")
    print(f"准确率：{report['accuracy']}")
    print(f"正确分类的样本数：{report['accuracy_num']}")
    print(f"微平均准确率：{report['micro_precision']}")
    print(f"微平均召回率：{report['micro_recall']}")
    print(f"微平均F1得分：{report['micro_f1']}")
    print(f"宏分类报告：\n{report['macro_class_report']}")

    # ============= 混淆矩阵 =============
    if cm_flag:
        plotConfusionMatrix(f"MSRE-GRU", report["confusion_matrix"], report["labels"], save_path)
