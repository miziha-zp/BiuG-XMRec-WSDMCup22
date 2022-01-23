
for l2 in  700 500
do 
python -u main.py --train_data ../ItemKNN/data/data/t2/train_enmf.txt --test_data ../ItemKNN/data/data/t2/test_enmf.txt --l2_reg $l2 --dataset t2
python -u main.py --train_data ../ItemKNN/data/data/t1/train_enmf.txt --test_data ../ItemKNN/data/data/t1/test_enmf.txt --l2_reg $l2 --dataset t1
python -u main.py --train_data ../ItemKNN/data/data/t1_testing/train_enmf.txt --test_data ../ItemKNN/data/data/t1_testing/test_enmf.txt --l2_reg $l2 --dataset t1_testing
python -u main.py --train_data ../ItemKNN/data/data/t2_testing/train_enmf.txt --test_data ../ItemKNN/data/data/t2_testing/test_enmf.txt --l2_reg $l2 --dataset t2_testing
done
