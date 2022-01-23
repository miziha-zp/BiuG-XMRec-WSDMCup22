
## ToDo
1. dataprocess
```bash
python xmrec_main.py
```


### 
1. python main.py --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --dataset="yelp2018" --topks="[10]" --recdim=64

 CUDA_VISIBLE_DEVICES=1 python main.py --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --dataset="t2" --topks="[10]" --recdim=128 

  CUDA_VISIBLE_DEVICES=2 python main.py --decay=1e-3 --lr=1e-3 --layer=4 --seed=2020 --dataset="t2" --topks="[10]" --recdim=128 --bpr_batch 4096 
  CUDA_VISIBLE_DEVICES=1 python main.py --decay=1e-4 --lr=1e-3 --layer=4 --seed=2020 --dataset="t1" --topks="[10]" --recdim=1024 --bpr_batch 4096 


CUDA_VISIBLE_DEVICES=2 python main.py --decay=1e-5 --lr=1e-3 --layer=4 --seed=2020 --dataset="t1" --topks="[10]" --recdim=64 --bpr_batch=4096 --comment 'decay3e-4' 688->681(overfitting)


CUDA_VISIBLE_DEVICES=2 python main.py --decay=1e-4 --lr=1e-3 --layer=4 --seed=2020 --dataset="t2" --topks="[10]" --recdim=2048 --bpr_batch=8192  > logs/lr1e_3.log 

CUDA_VISIBLE_DEVICES=1 python main.py --decay=1e-5 --lr=1e-4 --layer=4 --seed=2020 --dataset="t2" --topks="[10]" --recdim=2048 --bpr_batch=8192


### decay (only train5core)
CUDA_VISIBLE_DEVICES=1 python main.py --lr=1e-3 --layer=4 --seed=2020 --dataset="t2" --topks="[10]" --recdim=2048 --bpr_batch=8192 --decay=1e-3

--decay=1e-5  score(ndcg10_val)=0.458
--decay=1e-4  score(ndcg10_val)=0.576
--decay=1e-3  score(ndcg10_val)=0.563

CUDA_VISIBLE_DEVICES=3 python main.py --lr=1e-3 --layer=4 --seed=2020 --dataset="t2" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=2048 (train+train5core)

--decay=3e-5 score(ndcg10_val)=0.6043
--decay=5e-5 score(ndcg10_val)=0.6041
--decay=6e-5 score(ndcg10_val)=0.6045 （ok）
--decay=1e-4 score(ndcg10_val)=0.603
--decay=3e-4 score(ndcg10_val)=0.602
--decay=2e-4 score(ndcg10_val)=0.602


### train+train5core(ok)
CUDA_VISIBLE_DEVICES=2 python main.py --lr=1e-3 --layer=4 --seed=2020 --dataset="t2" --topks="[10]" --recdim=2048 --bpr_batch=8192 --decay=1e-4
(only train5core): score(ndcg10_val)=0.576
(train+train5core): 

### recdim
CUDA_VISIBLE_DEVICES=1 python main.py --lr=1e-3 --layer=4 --seed=2020 --dataset="t2" --topks="[10]"  --bpr_batch=8192 --decay=1e-4 --recdim=1024
--recdim=2048 score(ndcg10_val)=0.603
--recdim=1024 score(ndcg10_val)=0.602

### global data
CUDA_VISIBLE_DEVICES=1 python main.py --lr=1e-3 --layer=4 --seed=2020 --dataset="t2" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=2048

(only t1) score(ndcg10_val)=0.698
(t2+t1)   

### layer

CUDA_VISIBLE_DEVICES=3 python main.py --lr=1e-3 --layer=3 --seed=2020 --dataset="t1" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=2048

--layer=5  score(ndcg10_val)=0.696
--layer=4  score(ndcg10_val)=0.698
--layer=3  score(ndcg10_val)=0.692

### init
CUDA_VISIBLE_DEVICES=3 python main.py --lr=1e-3 --layer=4 --seed=2020 --dataset="t1" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=2048 --dropout=1 --keepprob=0.8
nn.init.normal_ : score(ndcg10_val)=0.698(ok)
nn.init.xavier_uniform_:score(ndcg10_val)=0.685

### model
CUDA_VISIBLE_DEVICES=2 python main.py --lr=1e-3 --layer=4 --seed=2022 --dataset="t1" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=2048 
CUDA_VISIBLE_DEVICES=3 python main.py --lr=1e-3 --layer=4 --seed=2022 --dataset="t2" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=2048 
CUDA_VISIBLE_DEVICES=3 python main.py --lr=1e-3 --layer=4 --seed=2020 --dataset="t2" --topks="[10]"  --bpr_batch=256 --decay=6e-5 --recdim=64  --model='mf'
### lr
CUDA_VISIBLE_DEVICES=3 python main.py  --layer=4 --seed=2022 --dataset="t1" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=2048 --lr=1e-3

--lr=1e-4 689
--lr=3e-4 699
--lr=1e-3 698
--lr=1e-2 (过山车660)

### 

 CUDA_VISIBLE_DEVICES=3  python main_seeds.py --dataset="t1" --topks="[10]"  --bpr_batch=40000 --decay=6e-5 --recdim=2048 --lr=1e-3 --onlytest


### pretrain - finetune
CUDA_VISIBLE_DEVICES=2  python pretrain.py --dataset="t1" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=64 --lr=1e-3 
CUDA_VISIBLE_DEVICES=2  python finetune.py --dataset="t1" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=64 --lr=1e-4 --load=1

 CUDA_VISIBLE_DEVICES=3  python main_seeds.py --dataset="t2" --topks="[10]"  --bpr_batch=8192 --decay=1e-3 --recdim=2048 --lr=1e-3 --epochs=1000 --model=mf

 CUDA_VISIBLE_DEVICES=3  python main_seeds.py --dataset="t2" --topks="[10]"  --bpr_batch=8192 --decay=5e-3 --recdim=2048 --lr=0.003 --epochs=2000 --model=mf
 CUDA_VISIBLE_DEVICES=3  python main_seeds.py --dataset="t1" --topks="[10]"  --bpr_batch=8192 --decay=5e-3 --recdim=2048 --lr=1e-3 --epochs=300 --model=lgn



## simcse
CUDA_VISIBLE_DEVICES=2  python main_seeds.py --dataset="t1" --dropout=1 --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=2048 --lr=1e-3 --epochs=250 --model=lgn


 CUDA_VISIBLE_DEVICES=2  python main_seeds.py --dataset="t2" --topks="[10]"  --bpr_batch=8192 --decay=6e-5 --recdim=1024 --lr=1e-3 --epochs=250 --model=lgn
