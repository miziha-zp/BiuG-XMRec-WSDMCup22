cd ../result/lgb_ranking
pwd
zip -r ../../learn2rank/submit1.zip ./
pwd
cd ../../learn2rank
pwd
python validate_submission.py --submission_file submit1.zip --data_dir ../input
