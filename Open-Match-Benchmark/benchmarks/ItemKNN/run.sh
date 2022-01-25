for k in 1 2 3 4 5 6 7  8 9 10 15 20
do 
python ItemKNN.py --K=$k --dataset t3
python ItemKNN.py --K=$k --dataset t3_testing
done
