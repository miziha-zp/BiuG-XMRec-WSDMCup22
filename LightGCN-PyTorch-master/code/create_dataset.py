import utils
import torch
import numpy as np
import pandas as pd
import time
import Procedure
from os.path import join
# ==============================
print(">>SEED:", 2020)
# ==============================
# import register
# from register import dataset


if __name__ == '__main__':
    utils.tokenizer('../../input/t1', 't1')
    utils.tokenizer('../../input/t2', 't2')
    
    utils.data_process('../../input/t1', 't1', training=False)
    utils.data_process('../../input/t2', 't2', training=False)

    utils.data_process('../../input/t1', 't1', training=True)
    utils.data_process('../../input/t2', 't2', training=True)
        
