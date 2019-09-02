# -*- coding: utf-8 -*
import sys

import os

# 训练数据的路径
TRAIN_DATA_PATH = os.path.join(sys.path[0], 'data', 'train')
CHECK_DATA_PATH = os.path.join(sys.path[0], 'data', 'test')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'model', 'saved_model')
TRAINED_MODEL_PATH = os.path.join(sys.path[0], 'model', 'pre_trained_model')
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')