#!/usr/bin/env python
# coding=utf-8
"""
@Description   开发环境配置
@Author        Alex_McAvoy
@Date          2025-10-01 16:27:33
"""

class DevConfig(object):
    """
    @description 开发环境配置
    """

    def __init__(self, main_tmp_path: str, dataset: str) -> None:
        """
        @description 构造函数
        @param {*} self 类实例化对象
        @param {str} tmp_path 主程序临时目录
        @param {str} dataset 采用数据集
        """
        # 路径配置
        tmp_path = "D:/Workspace/Python/EC-RE-GRU/temp"
        self.tmp_path = tmp_path
        if dataset == "USTC-TFC2016":
            tmp_path += "/USTC-TFC2016/"
        if dataset == "UNSW-NB15":
            tmp_path += "/UNSW-NB15/"
        tmp_path += main_tmp_path

        # 日志配置
        self.LOG = {
            # 相关文件保存路径
            "file_path": tmp_path + "/log/"
        }

        # 通用配置
        self.COMMON = {
            # 随机种子
            "seed": 36
        }

        # 采样配置
        self.SAMPLE = {
            # 有效样本数超参
            "beta": 0.9999,
            # 采样包最小长度
            "min_len": None,
            # 是否保存未采样数据
            "unsample_switch": False,
            # 相关文件保存路径
            "file_path": tmp_path + "/sample/",
        }

        self.WINDOW = {
            "l2_mode": "ethernet",
            # 窗口大小
            "window_size": 32,
            # 滑动步长
            "step_size": 8,
            # 相关文件保存路径
            "file_path": tmp_path + "/window/",
        }

        self.ANONYMOUS = {
            # 相关文件保存路径
            "file_path": tmp_path + "/anonymous/",
        }

        self.FIXED_LENGTH = {
            "n": 1518 * 8,
            # 相关文件保存路径
            "file_path": tmp_path + "/fix/",
        }

        self.ENTROPY = {
            "alphas": [1,2,3,4,5,6,7,8,9,10],
            "num_bins": 50,
            "alpha_weighting": True,
            # 相关文件保存路径
            "file_path": tmp_path + "/entropy/",
        }

        self.DISTRIBUTION = {
            "feature_dim": 39,
            "entropy_embed_dim": 64,
            # 相关文件保存路径
            "file_path": tmp_path + "/distribution/",
        }

        self.DATA = {
            # 相关文件保存路径
            "file_path": tmp_path + "/dataset/",
        }

        self.EVALUATE = {
            # 结果小数位数
            "decimal_places": 4,
            # 相关文件保存路径
            "file_path": tmp_path + "/evaluate/",
        }

        self.MODEL = {
            "batch_size": 8,
            "epochs": 20,
            "lr": 0.005,
            # 时间窗口长度 T
            "seq_len": 10,   
            "hidden_dim": 128,
            # 相关文件保存路径
            "file_path": tmp_path + "/model/",
        }

        if dataset == "USTC-TFC2016":
            # 数据集配置
            self.DATASET = {
                # 数据集名
                "name": "USTC-TFC2016",
                # PCAP包存储路径
                "pcap_dir": "D:/虚拟c盘/USTC-TFC2016/PCAPs_label/",
                # 流量类别映射字典
                "attack_type": {
                    0: "正常流量",
                    1: "Cridex",
                    2: "Geodo",
                    3: "Htbot",
                    4: "Miuref",
                    5: "Neris",
                    6: "Nsis-ay",
                    7: "Shifu",
                    8: "Tinba",
                    9: "Virut",
                    10: "Zeus",
                }
            }
            # 采样包最小长度
            self.SAMPLE["min_len"] = 60
            
            
        if dataset == "UNSW-NB15":
            # 数据集配置
            self.DATASET = {
                # 数据集名
                "name": "UNSW-NB15",
                # PCAP包存储路径
                "pcap_dir": "D:/虚拟c盘/UNSW-NB15/PCAPs_label/",
                # 流量类别映射字典
                "attack_type": {
                    0: "正常流量",
                    1: "Fuzzers",
                    2: "Analysis",
                    3: "Backdoors",
                    4: "DoS",
                    5: "Exploits",
                    6: "Generic",
                    7: "Reconnaissance",
                    8: "Shellcode",
                    9: "Worms",
                }
            }
            # 采样包最小长度
            self.SAMPLE["min_len"] = 40


        