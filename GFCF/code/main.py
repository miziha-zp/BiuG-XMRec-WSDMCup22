import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1
data_path={
    "t1":'../../input/t1',
    "t2":'../../input/t2',
    "t1_testing":'../../input/t1',
    "t2_testing":'../../input/t2',
      
} 

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    if(world.simple_model != 'none'):
        epoch = 0
        cprint("[TEST]")
        # Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        Procedure.predict(dataset, data_path[world.dataset], testing='test' in world.dataset)
finally:
    if world.tensorboard:
        w.close()
