# PU-LP-for-fake-news-detection
Positive and Unlabeled Learning by Label Propagation for Fake News Detection

This is an extension of the PU-LP algorithm (Ma 2017), which includes terms in the k-nn news network for fake news detection.

To run the framework, use ./run.sh passing two parameters: how many processes will run the representation model, the knn and W matrices; for each process, how many will perform the steps of pu-lp.
Example: ./run.sh 2 5
2 processes will perform the matrix calculation and 10 processes (2 * 5) will perform PU-LP and label propagation algorithms.

To modify the parameters of the algorithms, use the file create_scripts.py:
1. representation_model: text that indicates the name of the representation model used in the experiments.
2. arg_tsv_file: tsv file path organized in 3 columns (label, news and label - 1 for fake and -1 for non-fake)
3. arg_stopwords: .txt file containing domain stopwords.
4. arg_language: string informing the language of the news ('portuguese' or 'english').
5. arg_options: options (1 for stopwords removal and 2 for stopwords removal and radicalization).
6. arg_ngram_range: tfidfVectorizer.n_gram_range (1,1) for unigrams, (1,2) for unigrams and bigrams, (2,2) for bigrams.
7. arg_min_df: tfidfVectorizer.min_df ignores terms that have a document frequency strictly below the limit cut-off. If float, the parameter represents a proportion of documents.
8. arg_norm: tfidfVectorizer.norm (l1: sum of the absolute values ​​of the elements of the vector is 1. l2: sum of the squares of the elements of the vector is 1)
9. arg_min_weigth: #Min tf-idf value to remove terms of knn networks
10. arg_k: k values for k-nn matrix
11. arg_a: alpha parameter of pu-lp
12. arg_l: pu-lp lambda parameter
13. arg_m: parameter m of pu-lp
14. arg_mi: GNetMine alpha parameter
15. arg_max_iter: maximum number of iterations of the label propagation algorithm
16. agr_limiar_conv: convergence threshold of the label propagation algorithm
17. weight_relations: weight of relations by layer in the heterogeneous network
18. folds: cross validation folds

After executing ./run.sh, scripts in the "scripts" folder are executed. The results are saved in the "results" and "results/confusion" folder.

References:
1. Ma 2017: Ma, Shuangxun, and Ruisheng Zhang. "PU-LP: A novel approach for positive and unlabeled learning by label propagation." 2017 IEEE International Conference on Multimedia & Expo Workshops (ICMEW). IEEE, 2017.
2. Monteiro, Rafael A., et al. "Contributions to the study of fake news in portuguese: New corpus and automatic detection results." International Conference on Computational Processing of the Portuguese Language. Springer, Cham, 2018.
3. Shu, Kai, et al. "Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media." Big Data 8.3 (2020): 171-188. 
