# coding=UTF-8
"""
@Description   wrapper 装饰器
@Author        Alex_McAvoy
@Date          2023-11-22 09:06:10
"""
from functools import wraps

import time
from utils.log import Logger


def calTimes(logger: Logger, msg: str):
    """
    @description 计算函数运行时间
    @param {Logger} logger 日志控制器
    @param {str} msg 消息
    @return {*} dector 函数闭包
    """

    def dector(func):
        @wraps(func)
        def wrapper(*arg, **kwarg):
            s_time = time.time()
            res = func(*arg, **kwarg)
            e_time = time.time()

            if msg:
                logger.info(msg + "，耗时：%.2f s" % (e_time - s_time))
            else:
                logger.info("耗时：%.2f s" % (e_time - s_time))
            return res

        return wrapper

    return dector
