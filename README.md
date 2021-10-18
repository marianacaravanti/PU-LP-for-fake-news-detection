# Positive and Unlabeled Learning by Label Propagation for Fake News Detection

- Mariana Caravanti (ICMC/USP) | mariana.caravanti@usp.br
- Bruno Nogueira (Facom/UFMS) | bruno@facom.ufms.br
- Rafael Rossi (UFMS) | rafael.g.rossi@ufms.br
- Ricardo Marcacini | ricardo.marcacini@icmc.usp.br
- Brucce Neves | brucce.neves@usp.br
- Solange Rezende | solange@icmc.usp.br


# Abstract

Fake news can rapidly spread through internet users and can deceive a large audience. Due to those characteristics, they can have a direct impact on political and economic events. Machine Learning approaches have been used to assist fake news identification. However, since the spectrum of real news is broad, hard to characterize, and expensive to label data due to the high update frequency, One-Class Learning (OCL) and Positive and Unlabeled Learning (PUL) emerge as an interesting approach for content-based fake news detection using a smaller set of labeled data than traditional machine learning techniques. In particular, network-based approaches are adequate for fake news detection since they allow incorporating information from different aspects of a publication to the problem modeling. In this paper, we propose a network-based approach based on Positive and Unlabeled Learning by Label Propagation (PU-LP), a one-class and transductive semi-supervised learning algorithm that performs classification by first identifying potential interest and non-interest documents into unlabeled data and then propagating labels to classify the remaining unlabeled documents. A label propagation approach is then employed to classify the remaining unlabeled documents. We assessed the performance of our proposal considering homogeneous (only documents) and heterogeneous (documents and terms) networks.  Our comparative analysis considered four OCL algorithms extensively employed in One-Class text classification (k-Means, k-Nearest Neighbors Density-based, One-Class Support Vector Machine, and Dense Autoencoder), and another traditional PUL algorithm (Rocchio Support Vector Machine). The algorithms were evaluated in three news collections, considering balanced and extremely unbalanced scenarios. We used Bag-of-Words and Doc2Vec models to transform news into structured data. Results indicated that PU-LP approaches are more stable and achieve better results than other PUL and OCL approaches in most scenarios, performing similarly to semi-supervised binary algorithms. Also, the inclusion of terms in the news network activate better results, especially when news are distributed in the feature space considering veracity and subject. News representation using the Doc2Vec achieved better results than the Bag-of-Words model for both algorithms based on vector-space model and document similarity network.

# News Collections 
0
We evaluate the PUL and OCL algorithms considering balanced and unbalanced collections, news in Portuguese and in English, and collections containing only one subject or multiple subjects. The first collection was acquired from FakeNewsNet repository (https://github.com/KaiDMML/FakeNewsNet - Shu et al., 2020), which contains news of famous people fact-checked by the GossipCop (https://www.gossipcop.com/) website. The dataset has 5,298 real news and 1,705 fake news.  
FakeNewsNet is the collection with the greatest unbalance in the distribution of classes. 

The second collection, Fake.BR (https://github.com/roneysco/Fake.BR-Corpus), is the first reference corpus in Portuguese for fake news detection. The news was manually collected and labeled (Silva, et al. 2020). All of them have a textual format, available in their original sizes, and truncated. The truncation in the texts was carried out to have a text dataset with an approximate number of words, avoiding bias in the learning process. The corpus consists of 7,200 news items, distributed in 6 categories: politics 58%, TV and celebrities (21.4%), society and daily life (17.7%), science and technology (1.5%), economy (0.7%) and religion (0.7%). This corpus contains 3,600 fake news and 3,600 true news. 

The third news collection, also in Portuguese, was the result of a collection on fact-checking news - AosFatos (https://aosfatos.org/noticias/), Agência Lupa (https://piaui.folha.uol.com.br/lupa/), Fato ou Fake (https://g1.globo.com/fato-ou-fake/), UOL Confere (https://noticias.uol.com.br/confere/) and G1 - Política (https://g1.globo.com/politica/). The collection contains 2,168 news, in which 1,124 are real and 1,044 are fake, and was collected during our project's execution. Some terms that were added after the checking process were removed since they are correlated with the classes: "fato", "fake", "verdadeiro", "falso", "#fake", "verificamos", "montagem", "erro" e "checagem" (in English: fact, fake, real, check, and montage respectively).


# PU-LP

![methodology](https://github.com/marianacaravanti/PU-LP-for-fake-news-detection/blob/main/methodology.png)

# Features of each news Dataset

Summary of the news collections used in the experimental evaluation:

![datasets](https://github.com/marianacaravanti/PU-LP-for-fake-news-detection/blob/main/datasets.png)

# Results

Results using Bag-of-Words as representation model, comparing the proposed approach, the traditional PUL and OCL algorithms and the binary semi-supervised reference model for detecting fake news. HM denotes the use of homogeneous networks algorithms and HT denotes the use of heterogeneous networks algorithms. 10%, 20% e 30% indicate the percentage of fake news used in the labeled set. For our reference algorithm (BL), the labeled set also has the same percentage of real news. The best results considering PUL and OCL algorithms are highlighted in grey.

![results_bow](https://github.com/marianacaravanti/PU-LP-for-fake-news-detection/blob/main/results_BoW.png)

Results using Doc2Vec as representation model, comparing the proposed approach, the traditional PUL and OCL algorithms and the binary semi-supervised reference model for detecting fake news. HM are homogeneous networks algorithms and HT are heterogeneous networks algorithms. 10%, 20% e 30% indicate the percentage of fake news used in the labeled set. For our reference algorithm (BL), the labeled set also has the same percentage of real news. The best results considering PUL and OCL algorithms are highlighted in grey.

![results_d2v](https://github.com/marianacaravanti/PU-LP-for-fake-news-detection/blob/main/results_D2V.png)

Average ranking and standard deviation of the OCL, PUL and binary (BIN) algorithms, considering 10%, 20% and 30% of labeled data for the interest-F1 results. Last column presents the mean of the average rankings. The best performances considering PUL and OCL algorithms are highlighted in grey.

![average ranking](https://github.com/marianacaravanti/PU-LP-for-fake-news-detection/blob/main/average%20ranking.png)

# References
[PU-LP]: Ma, S., Zhang, R.: Pu-lp: A novel approach for positive and unlabeled learning by label propagation. In: 2017 IEEE International Conference on Multimedia & Expo
Workshops (ICMEW). pp. 537–542. IEEE (2017).

[FBR]: Silva, R.M., Santos, R.L., Almeida, T.A., Pardo, T.A.: Towards automatically filtering fake news in portuguese. Expert Systems with Applications 146, 113–199
(2020).

[FNN]: Shu, Kai, et al. Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media. Big data 8.3 (2020): 171-188.
