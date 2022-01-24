
# BiuG's solution to XMRec WSDM Cup2022
- [BiuG's solution to XMRec WSDM Cup2022](#biugs-solution-to-xmrec-wsdm-cup2022)
  - [introduction](#introduction)
  - [Evaluation](#evaluation)
  - [solution](#solution)
    - [benchmarking collaborative filtering method on single market](#benchmarking-collaborative-filtering-method-on-single-market)
      - [LightGCN(SIGIR20)[2]](#lightgcnsigir202)
      - [Matrix Factorization](#matrix-factorization)
      - [UltraGCN(CIKM21)[4]](#ultragcncikm214)
      - [GFCF(CIKM21)[7]](#gfcfcikm217)
      - [Open-Match-Benchmark](#open-match-benchmark)
      - [itemCF[8]](#itemcf8)
      - [userCF](#usercf)
      - [Popularity](#popularity)
    - [Bringing in useful information from the source market](#bringing-in-useful-information-from-the-source-market)
      - [xm-itemCF](#xm-itemcf)
      - [xm-userCF](#xm-usercf)
      - [xm-Popularity](#xm-popularity)
    - [learn to rank](#learn-to-rank)
    - [details](#details)
      - [concat train and train5core](#concat-train-and-train5core)
      - [add validation data to train for test scoring.](#add-validation-data-to-train-for-test-scoring)
  - [run our code](#run-our-code)
    - [clone the repository](#clone-the-repository)
    - [install python package.](#install-python-package)
    - [quick run](#quick-run)
    - [run from the start](#run-from-the-start)
  - [teammate](#teammate)
  - [references](#references)
[link](https://competitions.codalab.org/competitions/36050#learn_the_details)


## introduction
>
E-commerce companies often operate across markets; for instance, Amazon has expanded their operations and sales to 18 markets (i.e. countries) around the globe. The cross-market recommendation concerns the problem of recommending relevant products to users in a target market (e.g., a resource-scarce market) by leveraging data from similar high-resource markets, e.g. using data from the U.S. market to improve recommendations in a target market. The key challenge, however, is that data, such as user interaction data with products (clicks, purchases, reviews), convey certain biases of the individual markets. Therefore, the algorithms trained on a source market are not necessarily effective in a different target market.
Despite its significance, small progress has been made in cross-market recommendation, mainly due to a lack of experimental data for the researchers. In this WSDM Cup challenge, we provide user purchase and rating data on various markets, enriched with review data in different languages, with a considerable number of shared item subsets. The goal is to improve individual recommendation systems in these target markets by leveraging data from similar auxiliary markets.

in short, try to using the source market data improve the recommendation in target market.
## Evaluation
nDCG@10 follow the protocol mentioned in [1], recall top10 items from 100 candidate for both target market t1 and t2.
## solution
we introduce our solution using the three below. We first benchmark collaborative filtering method on single market, including a lots of the classic approaches as well as the advanced approaches. Then we try to fit some of these approaches to a cross-market use. For greater performance, we use learn2rank method to fusion all the scores mentioned.
### benchmarking collaborative filtering method on single market
in the section, we benchmark collaborative filtering method on single market, including a lots of the classic approaches as well as the advanced approaches. Thanks for the open source code from authors or other good guys.

#### LightGCN(SIGIR20)[2]

   LightGCN just remove the feature transformation and nonlinear activation in NGCF(Neural Graph Collaborative Filtering[3]), and get great improvement. We use the [pytorch implement](https://github.com/gusye1234/pytorch-light-gcn), and turn learning rate from [1e-3, 1e-4, 1e-2], l2_reg in [1e-2, 6e-5, 3e-4]and latent-dim from [64, 1024, 2048, 3072], layer in [2, 3, 4, 5, 6]. We found that the three hyperparameter significantly impact on performance. In our expriment, we set layer=4, learning_rate=1e-3, latent-dim=2048, l2_reg=6e-5 with the batchsize=8192, and then get the best single model performance(**NDCG@10=698 for t1, and NDCG@10=604 for t2, both of the result are validation result**).  For better performance, we blending five models with different initial seeds.
   **lightgcn is the best single model in our solution.**

#### Matrix Factorization
  
  Different from the bias awared MF mentioned in [6], we adapt a standard MF. and turn the hyperparameter like lightgcn. In our expriment, we set learning_rate=1e-3, latent-dim=2048, l2_reg=5e-3 with the batchsize=8192, and then get the best single model performance(NDCG@10=686 for t1, and NDCG@10=591 for t2, both of the result are validation result). For better performance, we blending five models with different initial seeds.

  **MF is slightly worse than lightgcn, but 2-3 faster than  lightgcn.**

  you can find more detail in [LightGCN-PyTorch-master/code](LightGCN-PyTorch-master/code), magic hyperparameter is hiden in `run_lgcn.sh` and `run_mf.sh`.

#### UltraGCN(CIKM21)[4]

  UltraGCN is ultra-simplified formulation of GCNs, which skips infinite layers of message passing for efficient recommendation. in the paper, the performance is amazing. But this too much hyper-parameter is a big problem for us.
  We turn the hyper-parameter that contained in mf, and find hard to get better result than mf.
#### GFCF(CIKM21)[7]

  [7]present a simple and computationally efficient method, named GF-CF. GFCF develop a generalgraph filter-based framework for CF, built upon the closed-form solution. we run the code from [author](https://github.com/yshenaw/GF_CF). The result is worse than mf.

#### Open-Match-Benchmark

  We run the four methods(EASE_r, ItemKNN, SLIM, Item2Vec) is from the collection **[Open-Match-Benchmark](https://openbenchmark.github.io/BARS/)**[6], many thanks for their work. For more detail, feel free to visit their [websit](https://openbenchmark.github.io/BARS/)

#### itemCF[8]

  ItemCF is classic collaborative filtering method, which recommend similar items similar to those purchased by the user.  In this competition, we adapt itemCF to a ranking use. For a pair(user, item) to be scored, we simply calculate the similarity between the item and items which the use purchased before,  then get a similarity sequence. We use the some the statistics for this sequence as the final score.

  For the similarity calculation between the two items, different ways are adapted by us. 
  - IOU(intersection of union between the users sequence of two items.)
  - cosine(cosine similarity between the users sequence of two items.)
  - cosine_item2vec(cosine similarity between the items vectors(which got from the item2vec[9]))

  For the statistics for this similarity sequence, different ways are adapted by us.
  - max, mean, std, median, length
  - 5%, 95% percentage

#### userCF
  
  Similar to ItemCF, UserCF recommend items purchased by the similar users to the user.  In this competition, we alse adapt UserCF to a ranking use. For a pair(user, item) to be scored, we simply calculate the similarity between the user and users whose purchased this item before,  then get a similarity sequence. We use the some the statistics for this sequence as the final score. The similarity and statistics is similar to itemCF.

#### Popularity
  
  Popular items are often popular, we calulation the click numbers of every item as the feature to represent the popularity of items.

**notice** The data use in this section, is only the target market data.


### Bringing in useful information from the source market

To alleviate the problem of sparse data in target markets, we make some attempts.

#### xm-itemCF
We simply add source market data to item similarity calculation, which means that when the users in the source market whose purchase the item are considered. After add source market data some scores are improved Obviously.
#### xm-userCF
Similar to xm-itemCF, we alse add users whose are from source market to purchase-items sequence.
#### xm-Popularity
 We calulation the click numbers of every item **in every market** as the feature to represent the popularity of items in different market.

### learn to rank
 Using the scoring method above, we can get the scoring of both validation data and test data, then we train a sorting model using validation data. LightGBM, short for Light Gradient Boosting Machine, is a free and open source distributed gradient boosting framework for machine learning originally developed by Microsoft[11].We adapter a LightGBM ranker for the sorting, which significantly improved performance.
 Follow [10], 7fold cross-validation is applied to get offline score.

An overall table of scores mentioned above is below. Due to space constraints， some scores are omitted. If you want all the score, see [here](#run-from-the-start)

for t1

| method             | ndcg@10validation | hit@10validation |
| ------------------ | ----------------- | ---------------- |
| lightgbm           | 0.725             | 0.826            |
| lightgcn           | 0.698             | 0.806            |
| mf_score           | 0.690             | 0.785            |
| ultraGCN           | 0.681             | 0.780            |
| gf-cf              | 0.675             | 0.761            |
| xm-itemcf-iou-mean | 0.686             | 0.782            |
| itemcf-iou-mean    | 0.677             | 0.761            |


for t2

| method             | ndcg@10validation | hit@10validation |
| ------------------ | ----------------- | ---------------- |
| lightgbm           | 0.632             | 0.749            |
| lightgcn           | 0.607             | 0.721            |
| mf_score           | 0.597             | 0.701            |
| ultraGCN           | 0.577             | 0.677            |
| gf-cf              | 0.556             | 0.650            |
| xm-itemcf-iou-mean | 0.586             | 0.685            |
| itemcf-iou-mean    | 0.566             | 0.663             |

for t1|t2

| method             | ndcg@10validation | hit@10validation |
| ------------------ | ----------------- | ---------------- |
| lightgbm           | 0.663             | 0.774            |

### details

#### concat train and train5core
We just concat train and train5core data and remove the duplicate sections as our train data.
#### add validation data to train for test scoring.
When score for test data, we add add validation groudtruth data to train.
## run our code  

### clone the repository

```bash
git clone https://github.com/miziha-zp/XMReC-WSDMCup-biuG.git
```
###  install python package.
```bash
pip install requirements.txt
```
### quick run
Due to "great size of the project", we highly recommend you to use our dumped features(which is in the attachment to email). For the total run, follow [run from the start](#run-from-the-start)
```bash
cd learn2rank
python 
```
### run from the start
set your work home in `run_all.sh`, and then,
```bash
bash run_all.sh 
```
if you want to get scores from every method mentioned above, motify `python xm_lgb.py` in `run_all.sh` to `python xm_lgb.py --offline`, then you will get the scores table.
your will find the result in `learn2rank/submit1.zip`.

notice: We rebuild our project for simplicity and rerun our project, so the result may be slightly different from the leaderboard, caused by the random factor.
if your have any problems to reproduce our solution, feel free to contact me(zzhangpeng@zju.edu.cn) with email.

## teammate
Peng Zhang from Zhejiang University， China. 

Rongxiu Gao from Zhejiang University，China. 

Changyv Li from University of Electronic Science and Technology of China.

Baoyin Liu from Northeastern University， China.

and our tutor Sheng Zhou, assistant professor at School of Software Technology, Zhejiang University, china.

Thank you to everyone for your efforts。

## references

[1] He et al. Neural Factorization Machines for Sparse Predictive Analytics

[2] He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

[3] He et al. Neural Graph Collaborative Filtering

[4] Mao et al. UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation

[5] Rendle et al. Neural Collaborative Filtering vs. Matrix Factorization Revisited

[6] Mao et al. SimpleX: A Simple and Strong Baseline for Collaborative Filtering

[7] Shen et al. How Powerful is Graph Convolution for Recommendation?

[8] Linden et al. Recommendations Item-to-Item Collaborative Filtering

[9] Barken et al. Item2Vec: Neural Item Embedding for Collaborative Filtering

[10] Chang et al. LIBSVM--A Library for Support Vector Machines

[11] Ke et al. LightGBM: A Highly Efficient Gradient Boosting
Decision Tree