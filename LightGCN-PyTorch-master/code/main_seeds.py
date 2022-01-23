import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
from save_utils import *
import register
from register import dataset
from validate_submission import *

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

valid_submit = []
test_submit = []
testing = 'testing' in world.dataset
for seed in [666, 2021, 333, 42, 3777]:# 
    # ==============================
    utils.set_seed(seed)
    layer = 4
    print(">>>>>>>>>>>>>>>>>>>>SEED:", seed)
    # ==============================
    world.config['lightGCN_n_layers'] = layer
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    # Recmodel.load_state_dict(torch.load("weight_file", map_location=torch.device('cpu')))
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)

    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        if epoch % 20 == 0:
            cprint(f"{layer}====[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            ids_valid_data, ids_test_data = Procedure.predict(Recmodel, data_path[world.dataset], epoch)
            
        if epoch %10 == 0:
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            torch.save(Recmodel.state_dict(), weight_file)
    description = 'lgcn_score' if world.model_name == 'lgn' else 'mf_score'
    ids_valid_data, ids_test_data = Procedure.predict(Recmodel, data_path[world.dataset], epoch, testing=testing, description=description)
    valid_submit.append(ids_valid_data)
    test_submit.append(ids_test_data)
# merge 
def merge_submit(submits, des=''):
    sub = submits[0]
    for s in submits[1:]:
        sub[des] += s[des]
    return sub

valid_final = merge_submit(valid_submit, f'{description}_score')
test_final = merge_submit(test_submit, f'{description}_score')
world.cprint('============>  final saving')

description = 'lgcn_score' if world.model_name == 'lgn' else 'mf_score'
print('*-'*30)
offline_scores(valid_final, f'{description}_score', data_path[world.dataset], world.dataset)
print('*-'*30)
Procedure.save(valid_final, test_final, data_path[world.dataset], description=description, testing=testing)

valid_final['score'] = valid_final[f'{description}_score']
test_final['score'] = test_final[f'{description}_score']
del valid_final[f'{description}_score'], test_final[f'{description}_score']

# save_path = f'../result/tmp{getstrtime()}/'
save_path = '../result/lightgcn/'
save_single(valid_final, test_final, world.dataset, save_path=save_path)
world.cprint(save_path)

if world.tensorboard:
    w.close()