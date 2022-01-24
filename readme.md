
# BiuG's solution to XMRec WSDM Cup2022
- [BiuG's solution to XMRec WSDM Cup2022](#biugs-solution-to-xmrec-wsdm-cup2022)
  - [introduction](#introduction)
  - [Evaluation](#evaluation)
  - [solution](#solution)
    - [benchmarking collaborative filtering method on single market](#benchmarking-collaborative-filtering-method-on-single-market)
    - [Bringing in useful information from the source market](#bringing-in-useful-information-from-the-source-market)
    - [learn to rank](#learn-to-rank)
      - [popularity features](#popularity-features)
      - [history insection statistic](#history-insection-statistic)
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

- LightGCN(SIGIR20)[2]
- 
   LightGCN just remove the feature transformation and nonlinear activation in NGCF(Neural Graph Collaborative Filtering[3]), and get great improvement. We use the [pytorch implement](https://github.com/gusye1234/pytorch-light-gcn), and turn learning rate from [1e-3, 1e-4, 1e-2], l2_reg in [1e-2, 6e-5, 3e-4]and latent-dim from [64, 1024, 2048, 3072], layer in [2, 3, 4, 5, 6]. We found that the three hyperparameter significantly impact on performance. In our expriment, we set layer=4, learning_rate=1e-3, latent-dim=2048, l2_reg=6e-5 with the batchsize=8192, and then get the best single model performance(**NDCG@10=698 for t1, and NDCG@10=604 for t2, both of the result are validation result**).  For better performance, we blending five models with different initial seeds.
   **lightgcn is the best single model in our solution.**

- Matrix Factorization
  
  Different from the bias awared MF mentioned in [6], we adapt a standard MF. and turn the hyperparameter like lightgcn. In our expriment, we set learning_rate=1e-3, latent-dim=2048, l2_reg=5e-3 with the batchsize=8192, and then get the best single model performance(NDCG@10=686 for t1, and NDCG@10=591 for t2, both of the result are validation result). For better performance, we blending five models with different initial seeds.

  **MF is slightly worse than lightgcn, but 2-3 faster than  lightgcn.**

  you can find more detail in [LightGCN-PyTorch-master/code](LightGCN-PyTorch-master/code), magic hyperparameter is hiden in `run_lgcn.sh` and `run_mf.sh`.

- UltraGCN(CIKM21)[4]

  UltraGCN is ultra-simplified formulation of GCNs, which skips infinite layers of message passing for efficient recommendation. in the paper, the performance is amazing. But this too much hyper-parameter is a big problem for us.
  We turn the hyper-parameter that contained in mf, and find hard to get better result than mf.
- GFCF(CIKM21)[7]

  [7]present a simple and computationally efficient method, named GF-CF. GFCF develop a generalgraph filter-based framework for CF, built upon the closed-form solution. we run the code from [author](https://github.com/yshenaw/GF_CF). The result is worse than mf.

- Open-Match-Benchmark

  We run the four methods(EASE_r, ItemKNN, SLIM, Item2Vec) is from the collection **[Open-Match-Benchmark](https://openbenchmark.github.io/BARS/)**[6], many thanks for their work. For more detail, feel free to visit their [websit](https://openbenchmark.github.io/BARS/)

- itemCF[8]

  ItemCF is classic collaborative filtering method, which recommend similar items similar to those purchased by the user.  In this competition, we adapt itemCF to a ranking use. For a pair(user, item) to be scored, we simply calculate the similarity between the item and items which the use purchased before,  then get a similarity sequence. We use the some the statistics for this sequence as the final score.

  For the similarity calculation between the two items, different ways are adapted by us. 
  - IOU(intersection of union between the users sequence of two items.)
  - cosine(cosine similarity between the users sequence of two items.)
  - cosine_item2vec(cosine similarity between the items vectors(which got from the item2vec[9]))

  For the statistics for this similarity sequence, different ways are adapted by us.
  - max, mean, std, median, length
  - 5%, 95% percentage

- userCF
  
  Similar to ItemCF, UserCF recommend items purchased by the similar users to the user.  In this competition, we alse adapt UserCF to a ranking use. For a pair(user, item) to be scored, we simply calculate the similarity between the user and users whose purchased this item before,  then get a similarity sequence. We use the some the statistics for this sequence as the final score. The similarity and statistics is similar to itemCF.
  
### Bringing in useful information from the source market


### learn to rank
 

an overall benchmark is below.

for t1
| method   | ndcg@10validation | hit@10validation |
| -------- | ----------------- | ---------------- |
| lightgcn | 0.698             | 0.806            |
for t2
|method|ndcg@10validation|hit@10validation|

#### popularity features
#### history insection statistic

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
