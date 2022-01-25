
for l2 in  700 500
do 
python -u main.py --train_data ../ItemKNN/data/data/t3/train_enmf.txt --test_data ../ItemKNN/data/data/t3/test_enmf.txt --l2_reg $l2 --dataset t3
python -u main.py --train_data ../ItemKNN/data/data/t3_testing/train_enmf.txt --test_data ../ItemKNN/data/data/t3_testing/test_enmf.txt --l2_reg $l2 --dataset t3_testing
done
