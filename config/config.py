#!/usr/bin/env python
# coding=utf-8
"""
@Description   基础配置类
@Author        Alex_McAvoy
@Date          2025-10-03 16:38:11
"""

import os
from config.dev import DevConfig

class Config(object):
    def __init__(self, env: str, main_tmp_path: str, dataset: str) -> None:
        """
        @description 构造函数
        @param {*} self 类实例对象
        @param {str} env 环境
        @param {str} main_tmp_path 主程序临时目录
        @param {str} dataset 采用数据集
        @param {str} model 采用模型
        """
        # 开发环境
        if env == "dev":
            self._config = DevConfig(main_tmp_path, dataset)

        self.main_tmp_path = main_tmp_path
        self.mkdir()

    @property
    def config(self):
        return self._config


    def mkdir(self):
        # 通用
        if not os.path.exists(self._config.LOG["file_path"]):
            os.makedirs(self._config.LOG["file_path"])

        # 采样实验配置
        if self.main_tmp_path == "our_experiment": 
            sample_path = self._config.SAMPLE["file_path"]
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            
            window_path = self._config.WINDOW["file_path"]
            if not os.path.exists(window_path):
                os.makedirs(window_path)

            anonymous_path = self._config.ANONYMOUS["file_path"]
            if not os.path.exists(anonymous_path):
                os.makedirs(anonymous_path)

            fix_path = self._config.FIXED_LENGTH["file_path"]
            if not os.path.exists(fix_path):
                os.makedirs(fix_path)
            
            entropy_path = self._config.ENTROPY["file_path"]
            if not os.path.exists(entropy_path):
                os.makedirs(entropy_path)

            distribution_path = self._config.DISTRIBUTION["file_path"]
            if not os.path.exists(distribution_path):
                os.makedirs(distribution_path)    

            data_path = self._config.DATA["file_path"]
            if not os.path.exists(data_path):
                os.makedirs(data_path)        

            model_path = self._config.MODEL["file_path"]
            if not os.path.exists(model_path):
                os.makedirs(model_path)       
            
            evaluate_path = self._config.EVALUATE["file_path"]
            if not os.path.exists(evaluate_path):
                os.makedirs(evaluate_path)   