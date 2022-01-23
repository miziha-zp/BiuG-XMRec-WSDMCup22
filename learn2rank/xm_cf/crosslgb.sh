python valid_lgb.py --t1_s t2 --t2_s t1 > logs/t2t1.log

python valid_lgb.py --t1_s s1 --t2_s s1 > logs/s1s1.log

python valid_lgb.py --t1_s s2 --t2_s s2 > logs/s2s2.log

python valid_lgb.py --t1_s s3 --t2_s s3 > logs/s3s3.log

python valid_lgb.py --t1_s s1,s2,s3,t2 --t2_s s1,t1 > logs/allin.log
