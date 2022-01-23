# baseline
python -u SLIM_main.py --dataset=t2 --topk=[10] --alpha=0.01 --l1=0.0001
Hyperparameters: alpha tuned from [1e-2, 2e-3, 1e-3, 5e-4, 1e-4], l1 tuned from [1e-3, 1e-4, 1e-5, 1e-6]

# l1 几乎无影响
# alpha -t1
0.01 0.669
0.03 0.673
0.04 0.674
0.05 0.673
0.08 0.668
0.1  0.664
0.002 0.64
0.0005 0.63

-t2

python -u SLIM_main.py --dataset=t2 --topk=[10] --alpha=0.05 --l1=1e-3
python -u SLIM_main.py --dataset=t2 --topk=[10] --alpha=0.04 --l1=1e-3
