#!/usr/bin/env python
# coding=utf-8
"""
@Description   only_Entropy GRU 模型评估
@Author        lazypanda666
@Date          2026-04-18 18:08:56
"""


import os
import torch
import numpy as np
from utils.log import logger
import sklearn.metrics as sm
import sklearn.preprocessing as sp
from evaluate.draw import plotConfusionMatrix
import model.gruOnlyEntropyAblation as gru   

from const import cfg
from utils.wrapper import calTimes


def _saveEvaluateResult(filename_suffix: str, res: dict, save_path: str) -> None:
    """
    @description 将评估结果写入文本文件
    @param filename_suffix: 文件名后缀
    @param res: 评估结果字典
    @param save_path: 保存路径
    @return {None}
    """      
    with open(save_path + f"{filename_suffix}_report.txt", "w", encoding="UTF-8") as f:
        f.write(f"{filename_suffix} 评估结果\n")
        f.write("--------------------------------------------\n")

        f.write(f"准确率：{res['accuracy']}\n")
        f.write(f"正确分类的样本数：{res['accuracy_num']}\n")
        f.write(f"微平均准确率：{res['micro_precision']}\n")
        f.write(f"微平均召回率：{res['micro_recall']}\n")
        f.write(f"微平均F1得分：{res['micro_f1']}\n")
        f.write(f"Macro PR-AUC：{res['pr_auc_macro']}\n")
        f.write(f"Macro ROC-AUC：{res['roc_auc_macro']}\n")
        f.write(f"宏分类报告：\n{res['macro_class_report']}\n")


def computeMacroCurves(y_test: np.ndarray, y_score: np.ndarray) -> dict:
    """
    @description 计算多分类任务下的指标
    @param y_test: 真实标签
    @param y_score: 预测概率
    @return dict:
        precision_macro: 平均 PR 曲线
        recall_common: 统一 recall 采样点
        pr_auc_macro: 宏平均 PR-AUC
        fpr_macro: 统一 FPR 采样点
        tpr_macro: 平均 ROC 曲线
        roc_auc_macro: 宏平均 ROC-AUC
    """
    unique_labels = np.unique(y_test)
    y_test_bin = sp.label_binarize(y_test, classes=unique_labels)
    num_classes = y_test_bin.shape[1]

    precision_ls, recall_ls = [], []
    fpr_ls, tpr_ls = [], []

    for i in range(num_classes):
        p, r, _ = sm.precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        precision_ls.append(p)
        recall_ls.append(r)

        fpr, tpr, _ = sm.roc_curve(y_test_bin[:, i], y_score[:, i])
        fpr_ls.append(fpr)
        tpr_ls.append(tpr)

    recall_common = np.linspace(0, 1, 200)
    precision_interpolated = np.zeros_like(recall_common)

    for p, r in zip(precision_ls, recall_ls):
        precision_interpolated += np.interp(recall_common, r[::-1], p[::-1])

    precision_macro = precision_interpolated / num_classes
    pr_auc_macro = sm.auc(recall_common, precision_macro)

    fpr_common = np.linspace(0, 1, 200)
    tpr_interpolated = np.zeros_like(fpr_common)

    for fpr, tpr in zip(fpr_ls, tpr_ls):
        tpr_interpolated += np.interp(fpr_common, fpr, tpr)

    tpr_macro = tpr_interpolated / num_classes
    roc_auc_macro = sm.auc(fpr_common, tpr_macro)

    return {
        "precision_macro": precision_macro,
        "recall_common": recall_common,
        "pr_auc_macro": pr_auc_macro,
        "fpr_macro": fpr_common,
        "tpr_macro": tpr_macro,
        "roc_auc_macro": roc_auc_macro
    }


@calTimes(logger, "评估完成")
def evaluateModel(
    attack_type: dict = cfg.DATASET["attack_type"],
    decimal_places: int = cfg.EVALUATE["decimal_places"],
    cm_flag: bool = True,
    save_path: str = cfg.EVALUATE["file_path"]) -> None:
    """
    @description 对 Entropy-only GRU 模型进行完整评估
    @param attack_type: 类别标签映射 dict
    @param decimal_places: 结果保留小数位数
    @param cm_flag: 是否绘制并保存混淆矩阵
    @param save_path: 评估结果保存路径
    @return {None}
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    E, DE, y_true = gru.loadSequenceData("test")

    E = torch.tensor(E, dtype=torch.float32).to(device)
    DE = torch.tensor(DE, dtype=torch.float32).to(device)
    y_true = torch.tensor(y_true, dtype=torch.long).to(device)


    model_path = os.path.join(
        cfg.MODEL["file_path"],
        "GRU_only_entropy.pth"
    )

    model = gru.EntropyGRU(
        entropy_dim=E.shape[-1],
        hidden_dim=cfg.MODEL["hidden_dim"],
        num_classes=len(attack_type)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    attack_type = {int(k): v for k, v in attack_type.items()}
    labels = sorted(attack_type.keys())
    attack_labels = [attack_type[i] for i in labels]


    with torch.no_grad():
        logits = model(E, DE)
        probs = torch.softmax(logits, dim=1)

        y_pred = torch.argmax(probs, dim=1).cpu().numpy()
        y_score = probs.cpu().numpy()
        y_true = y_true.cpu().numpy()

    accuracy = round(sm.accuracy_score(y_true, y_pred), decimal_places)
    accuracy_num = sm.accuracy_score(y_true, y_pred, normalize=False)

    macro_class_report = sm.classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=attack_labels,
        digits=decimal_places,
        zero_division=0
    )

    micro_precision = round(sm.precision_score(y_true, y_pred, average="micro"), decimal_places)
    micro_recall = round(sm.recall_score(y_true, y_pred, average="micro"), decimal_places)
    micro_f1 = round(sm.f1_score(y_true, y_pred, average="micro"), decimal_places)

    cm = sm.confusion_matrix(y_true, y_pred, labels=labels)

    curve_res = computeMacroCurves(y_true, y_score)

    report = {
        "labels": attack_labels,
        "accuracy": accuracy,
        "accuracy_num": accuracy_num,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_class_report": macro_class_report,
        "confusion_matrix": cm,
        "pr_auc_macro": round(curve_res["pr_auc_macro"], decimal_places),
        "roc_auc_macro": round(curve_res["roc_auc_macro"], decimal_places)
    }

    _saveEvaluateResult("GRU_only_entropy", report, save_path)

    print("Entropy-GRU 评估结果")
    print(f"准确率：{accuracy}")
    print(f"正确分类样本数：{accuracy_num}")
    print(f"微F1：{micro_f1}")
    print(f"Macro PR-AUC：{report['pr_auc_macro']}")
    print(f"Macro ROC-AUC：{report['roc_auc_macro']}")
    print(f"分类报告：\n{macro_class_report}")

    if cm_flag:
        plotConfusionMatrix("GRU_only_entropy", cm, attack_labels, save_path)