CODEHOME=/home/lichangyv/miziha/code_review/
RANKPATH=$CODEHOME/learn2rank/
LGBPATH=$CODEHOME/LightGCN-PyTorch-master/code
GFCFPATH=$CODEHOME/GFCF/code/
ultraGCNPATH=$CODEHOME/ultraGCN
EASE_RPATH=$CODEHOME/Open-Match-Benchmark/benchmarks/EASE_r
ItemKNNPATH=$CODEHOME/Open-Match-Benchmark/benchmarks/ItemKNN
SLIMPATH=$CODEHOME/Open-Match-Benchmark/benchmarks/SLIM
Item2VecPATH=$CODEHOME/Open-Match-Benchmark/benchmarks/Item2Vec
cd $LGBPATH
pwd
python create_dataset.py
bash run_mf.sh # run mf
bash run_lgcn.sh # run lightgcn
cd $GFCFPATH
pwd
bash run.sh
cd $ultraGCNPATH
bash run.sh
pwd
cd $EASE_RPATH
pwd
bash run.sh
cd $ItemKNNPATH
pwd
bash run.sh
cd $SLIMPATH
pwd
bash run.sh
cd $Item2VecPATH
pwd
bash run.sh
cd $RANKPATH
pwd
python xm_lgb.py 
bash offline_run.sh

