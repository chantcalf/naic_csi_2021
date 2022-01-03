#!/usr/bin/env python
# _*_coding:utf-8_*_
"""
@Time   :  2022/1/1 18:00
@Author :  Qinghua Wang
@Email  :  597935261@qq.com
"""
import logging
import os
import sys
from logging import handlers

CONFIG_DIR = os.path.realpath(os.path.dirname(__file__))


class Logger:
    """

    """
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=15,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


LOG_DIR = os.path.join(CONFIG_DIR, "../log")
os.makedirs(LOG_DIR, exist_ok=True)
TRAIN_DATA_DIR = os.path.join(CONFIG_DIR, "../train")
NUM_WORKERS = 4 if sys.platform == "linux" else 0  # WINDOWS下0比任意数都快
