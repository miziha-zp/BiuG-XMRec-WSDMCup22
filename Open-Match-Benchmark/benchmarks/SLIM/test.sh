python -u SLIM_main.py --dataset=t2 --topk=[10] --alpha=0.01 --l1=1e-3 > logs/alpha3.log
python -u SLIM_main.py --dataset=t2 --topk=[10] --alpha=0.03 --l1=1e-3 > logs/alpha4.log
python -u SLIM_main.py --dataset=t2 --topk=[10] --alpha=0.05 --l1=1e-3 > logs/alpha5.log
python -u SLIM_main.py --dataset=t2 --topk=[10] --alpha=0.08 --l1=1e-3 > logs/alpha6.log
# python -u SLIM_main.py --dataset=t2 --topk=[10] --alpha=1e-4 --l1=1e-3 > logs/alpha7.log
