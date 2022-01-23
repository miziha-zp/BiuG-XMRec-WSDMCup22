import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in world.all_dataset:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
    # try:
    #     finetune_dataset = dataloader.Loader(path="../data/"+world.dataset, finetune=True)
    # except Exception as e:
    #     print(e)
    #     pass
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}