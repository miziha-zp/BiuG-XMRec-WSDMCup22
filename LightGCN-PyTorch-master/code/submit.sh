cd ../result/lightgcn
zip -r ../../code/submit1.zip ./
pwd
cd ../../code
pwd
python validate_submission.py --submission_file submit1.zip --data_dir ../../input
