### decay (only train5core)
CUDA_VISIBLE_DEVICES=0 python solve.py --config-file='t2.ini' --dir ../input/t2
CUDA_VISIBLE_DEVICES=1 python solve.py --config-file='t1.ini' --dir ../input/t1

### K
K=40 685
K=20 684
K=100 682

### gamma
gamma = 

### neg_number

### neg_weight

python solve.py --dir="../input/t1" --config-file="t1.ini" >> logs/long.log

### lambda
lambda=2.75 685
lambda=5  683
lambda=0.1