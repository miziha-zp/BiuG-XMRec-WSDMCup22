import utils
# import torch
# import numpy as np
# import pandas as pd
# import time
# import Procedure
# from os.path import join
# ==============================
print(">>SEED:", 2020)
# ==============================
# import register
# from register import dataset


if __name__ == '__main__':
    utils.tokenizer('../../input/t3', 't3')
    utils.data_process('../../input/t3', 't3', training=False)
    utils.data_process('../../input/t3', 't3', training=True)
        
