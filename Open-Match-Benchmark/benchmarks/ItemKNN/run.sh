for k in 1 2 3 4 5 6 7  8 9 10 15 20
do 
python ItemKNN.py --K=$k --dataset t2
python ItemKNN.py --K=$k --dataset t1
python ItemKNN.py --K=$k --dataset t2_testing
python ItemKNN.py --K=$k --dataset t1_testing
done
