for k in 2048 256
do 
    CUDA_VISIBLE_DEVICES=2 python Item2Vec.py --dataset t3 --emb-dim=$k
    CUDA_VISIBLE_DEVICES=2 python Item2Vec.py --dataset t3_testing --emb-dim=$k
done