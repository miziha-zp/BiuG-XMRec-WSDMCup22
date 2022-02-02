for k in 2048 256
do 
    CUDA_VISIBLE_DEVICES=0 python Item2Vec.py --dataset t1 --emb-dim=$k
    CUDA_VISIBLE_DEVICES=0 python Item2Vec.py --dataset t2 --emb-dim=$k
    CUDA_VISIBLE_DEVICES=0 python Item2Vec.py --dataset t1_testing --emb-dim=$k
    CUDA_VISIBLE_DEVICES=0 python Item2Vec.py --dataset t2_testing --emb-dim=$k
done
