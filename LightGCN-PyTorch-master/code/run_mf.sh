 CUDA_VISIBLE_DEVICES=0  python main_seeds.py --dataset="t1" --topks="[10]"  --bpr_batch=8192 --decay=5e-3 --recdim=2048 --lr=1e-3 --epochs=400 --model=mf
 CUDA_VISIBLE_DEVICES=0  python main_seeds.py --dataset="t2" --topks="[10]"  --bpr_batch=8192 --decay=5e-3 --recdim=2048 --lr=1e-3 --epochs=350 --model=mf
 CUDA_VISIBLE_DEVICES=0  python main_seeds.py --dataset="t1_testing" --topks="[10]"  --bpr_batch=8192 --decay=5e-3 --recdim=2048 --lr=1e-3 --epochs=400 --model=mf
 CUDA_VISIBLE_DEVICES=0  python main_seeds.py --dataset="t2_testing" --topks="[10]"  --bpr_batch=8192 --decay=5e-3 --recdim=2048 --lr=1e-3 --epochs=350 --model=mf
 