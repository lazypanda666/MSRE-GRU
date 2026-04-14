#!/usr/bin/env python
# coding=utf-8
"""
@Description   配置文件
@Author        Alex_McAvoy
@Date          2025-09-19 14:25:25
"""

import os
import sys
import re

from config.config import Config


# ================================ 不同主函数中的内部配置 ================================
# 获取当前运行的主模块名
main_file_name = os.path.basename(sys.argv[0])
match = re.search(r'_(.*?)\.', main_file_name)
if match:
    name = match.group(1)

# 采样实验配置
if name == "our":
    # ============ 主程序临时目录 ============
    main_tmp_path = "our_experiment"
    

# ================================ 通用配置 ================================
# ============ 环境配置 ============
# 开发环境
env = "dev"
# 西理环境
# env = "prod_xaut"

# ============ 数据集配置 ============
# USTC-TFC2016数据集
dataset = "USTC-TFC2016"
# UNSW-NB15数据集
# dataset = "UNSW-NB15"


cfg = Config(env, main_tmp_path, dataset).config