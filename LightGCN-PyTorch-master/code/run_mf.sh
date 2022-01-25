 CUDA_VISIBLE_DEVICES=0  python main_seeds.py --dataset="t3" --topks="[10]"  --bpr_batch=8192 --decay=5e-3 --recdim=2048 --lr=1e-3 --epochs=400 --model=mf
 CUDA_VISIBLE_DEVICES=0  python main_seeds.py --dataset="t3_testing" --topks="[10]"  --bpr_batch=8192 --decay=5e-3 --recdim=2048 --lr=1e-3 --epochs=400 --model=mf
 