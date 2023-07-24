import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import paddle.vision.transforms as T
import math
import os
import sys 
sys.path.append('/home/aistudio/external-libraries')

import medpy.metric.binary as mmb
import work.newnet as newnet
import work.unet as unet
import work.functions as functions

nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant())
paddle.seed(42)
np.random.seed(42)

segmentor=unet.UNet()
train(100,lr=0.0003,resume_epoch=0)
