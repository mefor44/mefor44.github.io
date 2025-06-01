---
layout: post
title: "The Hidden Magic Behind Personalized Recommendations: Optimizing the Retrieval Stage"
author: Mateusz Marzec
date: 2024-12-04
tags: retrieval, advanced
---


## Table of Contents
1. [Introduction](#introduction)
2. [How / where are these models utilized?](#how--where-are-these-models-utilized)
3. [Architecture and feature choices](#architecture-and-feature-choices)
    1. [Meta](#meta)
    2. [JD](#jd)
    3. [Etsy](#etsy)
    4. [Ebay](#ebay)
6. [Training details](#training-details)
7. [Serving](#serving)
8. [Summary](#summary)
9. [References](#refences)




## Introduction
In this blog post I will discuss how some of the most known companies are making their "retrieval" more personalized. In a typical large scale industry setting, the amount of items and users is often humongous, which forces developers to design recommendation engines as a cascade of models rather than one standalone model. This can look like diagram below:

<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/cascade_ranking_system.png" width="150" />
</div>
We start at full pool of items, and at each stage we trim by selecting and ranking candidates, better and better. Not all steps are a must. Technically, only retrieval is completely necessary, as it is required by system requirements (like latency contrains).

For purpose of this post lest's take a closer look how companies like Meta (Facebook), JD, Pinterest, Etsy and Ebay introduce personalization into their systems - while focusing on retrieval stage only.
TODO: add logos of aftermentioned companies into one picture




## How / where are these models utilized?
In most cases they are used in search (JD, Etsy, Meta). For Ebay, they use model to create personalized carousels e.g. "sponsored products based on items you recently viewed". For Pinterest, they indirectly suggested that model outputs are used in many downstream task (mainly ranking models). Their user embeddings can be also utlized to retrieve relevant items for places like Pinterest Homefeed. In Meta paper they mention developing the same model for different subdomain of search (groups search, people search, page search).

## Architecture and feature choices
In most cases it is some variant of Two-tower model. It is a very useful paradigm in retrieval due to few reasons. Main it decupled query and item towers. This is a major thing - during inference we can pass query features through query tower and compute candidates by berforming efficient Approximate Nearest Neighbors Search. But firstly let's see how this architecture can look in practice.
### Meta
<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Meta-model-architecture.png" width="400" />
</div>
The towers are assymetric - Query encoder utilizes user features - demografic featuers, as well as location and social connections to help better understand overall query context. Document encoder uses features more tailored to specific use case. For example for a group searchprpoblem  aggregated location and social clusters of groups can be used. Categorical featues are encoded as embeddings (common theme). 
Architectures from Etsy, JD and Ebay follow similar pattern.


### JD
<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/JD-model-architecture.png" width="800" />
</div>
For JD we have query/user and (target) item towers. For item tower they use features like title, brand and category. For query tower they utilize query tokes, user profile and aggregated history of user events. One extention they use is mulit-head query tower. For $$ K $$ heads, input features are projected into $$ K $$ different spaces and separately transformed with $$ K $$ different MLPs. These product $$ K $$ output emebddings instead of usual one. This is meant to catch different intentions of input query. A vivid example is query "apple". Figure below shows a t-SNE visualizations of retrieval results for an example polysemous query.

<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/JD-2head-query-tower-tsne-apple.png" width="700" />
</div>
With just one head model is not able to distinguish between possible intents (Apple products vs a fruit). With two heads it clearly does better job. Of course adding addional heads comes with increase complexity, so it's worth investigating whether befefits outweights costs.

### Etsy
For Etsy model we have query and product encoders (yet another convention ;)).
<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Etsy-model-architecture.png" width="800" />
</div>
Product encoder utilises various features. Often they utilize additional models to transform some of the input features (transformer encoder for titles or graph encoder to encode neighbour information). This means they use components requiring heavy computation. This is not an issue as long as these components are in product encoder, which is used to build index only (not during online serving, thus inference doesn't have to be that fast). They use both location and token features for both query and product side. Token and location encoders are therefore shared between both towers. "Using all location features together, we observed a relative gain of 8% in purchase recall for domestic users.", which underlines the importance of using different set of available features. Scalar features are transfored to quantiles for both towers. For query tower recent user history (searches, clicks, etc.) is utilized to enrich query context. Lightweight transformer is used to produce historical embeddings for provided sequence of user-query vectors. "1-layer transformer contributes little to latency, and improves offline recall by 2%." Lastly, cosine similarity is used as a similarity measue to compute final similarity score between given query and product.

### Ebay

<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Ebay-model-architecture.png" width="800" />
</div>
To see more in-depth explanation of this architecture see [this post]({% post_url 2024-10-01-personalized-retrieval-ebay %}) about personalized recommendations at Ebay.

### Pinterest

In this section word "Pins" is used interchangebly with word "item". Authors of Pinnerformer list few design choices. Firstly, just one embedding for each user. In their prevous work [PinnerSage](https://arxiv.org/pdf/2007.03634), they didn't limit the number of embedding one user can get, which turns out to be a bit troublesome then using this embedding for downstreak tasks (like ranking). Second design choice is going for offline inference. Most work on sequential user modelling tries to achieve (near) real time inference which is inseparable with high computational costs, and infrastructure complexity. Ultimately they infer their model daily, balancing between performance and solution simplicity.

Their model tries to learn long-term user preferences based on engagement signals from Pinterest Homefeed page. 
<div style="text-align:center;">
<i>Our primary objective is to learn a model that is able to predict a user's positive future engagement over a 14 day time window after the generation of their embedding.</i>
</div>
<p></p>
It is quite an anusual task, in traditional sequence modelling it is more common to train model on "next action prediction" task. Considered positive user actions are: Pin save, a Pin close-up lasting over 10s, or a long clickthrough (>10s) to the link underlying a Pin. Based on sequence of positive user actions, the aim is to learn user representation that well alignes with target Pin representations. Pin representaions are obtained by PinSage model (addional trainable MLP is added at the end). PinSage is a graph-based neural network designed to generate high-quality embeddings for Pins by aggregating visual, text, and engagement data.

Component for user modelling constitutes of a transformer with casual masking layer followed by MLP block. Training window is 28 days (it exceeds evaluation window, which is equal to 14 days). See image below.

<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Pinterest-model-architecture.png" width="700" />
</div>

Each action in user sequence consist of 256 dimensional Pin embedding (from PinSage), and some metadata features: action type, surface, timestamp, and action duration. Categorical features are encoded as a lookup tabls, action duration is transformed via $$log$$ operator. Time features are encoded with sine and cosine transformations. All features are concatenated into one vector of dimension $$D_{in}$$.



## Training details
In this section let's take a glimpse into loss functions, methods of mining negatives and other things realted to training.
### Loss function
JD, Etsy and Meta use [max margin triplet losses](https://www.d2l.ai/chapter_recommender-systems/ranking.html#hinge-loss-and-its-implementation) (potentiall with some quirks). Intuition behind this loss it to separate positive pair from engative pair by a given distance margin. It can be formalized as follows:

$$
\mathcal{L} = \max(0, m - f(q, d^+) + f(q, d^-))
$$

where:
- $$f(q, d)$$ is the scoring function (two-tower model) that measures the relevance of document $$d$$ with respect to query $$q$$.
- $$ d^+ $$ is a relevant document.
- $$ d^- $$ is an irrelevant document.
- $$ m $$ is the margin.

For max margin triplet loss it is important to tune margin hyperparamter. 
<p style="text-align:center;">
<i>We found that tuning margin value is important – the optimal margin value varies a lot across different training tasks, and different margin values result in 5-10% KNN recall variance.</i>
</p>

Ebay uses [samplex softmax loss](https://www.linkedin.com/pulse/sampled-softmax-loss-retrieval-stage-recommendation-khrylchenko-6ngmf/). Intuition behind this loss funtion it to aviod computing whole softmax (the pool of items is massive), and compute it only on subset of items (approximation).Both loss functions them require negative instances, which are not necessarily present in data in context of retrieval problems.

In PinnerFormer authors use sampled softmax with logQ correction as a loss function. It introduces a correction that accounts for the fact that some negatives can be more or less probable under the sampling distribution (distribution under which we sample negative instances), compared to true distribution.
They also explore different training objectives:
<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Pinterest-training-objectives.png" width="600" />
</div>
Naive *next action prediciton* - simply prediciton next user action. [SasRec](https://arxiv.org/abs/1808.09781) (transformer based sequence based recommender) model, extend this idea and tries to predict next user action at every time step. In *all action prediction* the aim is to predict **all** actions (a sample for computational tracebility) a user will take in the next $$K$$ days, using his last available embedding. This forces the model to learn longer term interests. In *dense all action prediction* firstly sample few timestamps, and then for each timestamp try to predict a randomly select positive action that happend up to $$K$$ days after the selected timestamp. This idea borrows from SasRec.

### Negatives
Mining negatives is an essential problem for retrieval systems. In most cases training data is composed from user logs. Search queries are matched with clicked products creating positive training instances. But still we lack negatives. Common method of solving this problem is [Mixed Negative Sampling](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/) - a simple, yet effective techinique for efficient negative retrieval. It is a mix of in-batch negatives and totaly random negatives. Both JD and Pinterest use Mixed Negative Sampling. In Ebay they use in-batch negatives, with some twist. They pick "viewed but non clicked" items associated with other items in the batch. Etsy uses a combination of Uniform (random) negatives, in-batch negatives and negatives geenrated with [STAR](https://arxiv.org/abs/2104.08051) method. For Meta random negatives were "too easy", so they employed online negative mining strategy. For each query docuemnt they look for $$ k $$ most similar documents in the same batch and treat them as hard negatives. They also experimented with offline negative mining (selecting negatives from whole pool of items/documents). In this approach they compute similarity for all query documents present in training data and assign some number of hard negatives for each document (via ANN). A crucial observation is that using most similar documents as negatives degradates model performance. Using documents ranked $$ 100:500 $$ led to best model recall. An interesting quote:

<p style="text-align:center;">
<i>Mixed easy/hard training: blending random and hard negatives in training is advantageous. Increasing the ratio of easy to hard negatives continues to improve the model recall and saturated at easy:hard=100:1.</i>
</p>
highlights that ultimately their solution follows some form of Mixed Negative Sampling. Very useful concent to learn about! 

For Pinnerformer paper authors performed experiments to access hwo different sampling approaches work.
<div style="text-align:center;">
  <img src="/resources/general-personalized-retrieval/Pinterest-ablation-negatives.png" width="400" />
</div>
<p></p>
SPC stands for sample probability correction, P@90 Coverage is a diversity metric (described in [results section](#results)), and recall@10 is their main evaluation measure. Undoubtely the best recall is obatained for mixed sampling with SPC. Interestingly when using only random negatives, model fails to learn diverse user interests, and recommends non-diverse items.

## Serving
Paradigm for serving doesn't really change with introducing user component (compared to non-personalized retrieval). Diagram below present how whole system can look like (based on JD's system).

<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/JD-serving-architecture.png" width="400" />
</div>
In personalized scenario we additionaly need to take user account into consideration. This means more features to handle, but in most cases we should be able to just fetch them from feature store like any other feature. 
There are some optimizations possible when dealing with additional source of data. In JD they customized Tensorflow Dataset class to not duplicate item and user features - instead load them into memory and join "on-the-fly" when performing the training. This led to reducing data size by 90%. Another important thing to keep in mind is **Consistency Between Online and Offline**.

<p style="text-align:center;">
<i>A typical inconsistency usually happens at the feature computation stage, especially if two separate programming scripts are used in onine data preprocessing and online serving system.</i>
</p>

This is particulary diffucult as both stages requires different type of processing. For offline we need processing that can parallelize well to process multiple training instances at once. For serving we need processing that can handle single requests with very low latency. To alleviate this problem JD implemented tokenizer in pure C++. For offline data vocabulary computation they wrapped it with Python interface, and for online (and training) with Tensorflow custom C++ operator.

### JD's online serving system
Authors of JD's paper wanted to avoid having two separate online services for computing query embedding and search for nearest neighbors. They used Tensorflow servable framework which allowed them to unifity these two components into one model. With this approach query embedding is send to query emebdding index via computer memory (instead of computer network). This reduces chances of failure (for instance due to mapping mistake). Architecture is presented on image below

<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/JD-serving-architecture2.png" width="500" />
</div>
They additionally used model sharding to support using hundreds of models online at the same time. A proxy module which redirects model prediction requets to one of the servers holding corresponding model.


### Etsy's serving architecture
In Esty product and query vectors are expaded with additional features (for more details see [this section](#ann-based-product-boosting)). General architecture is similar to one from JD and is shown on image below.
<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Etsy-serving-architecture.png" width="700" />
</div>
Some of their observations include plenty of experiments with different ANN variants (such as HNSW, IVFlat, IVFPQFastScan from Faiss library). Ultimately using quantized index with additional re-ranking step led to 4% of recall loss with P99 latency under 20ms in production. P99 refers to 99th percentile of latency measurements. It's a performance metric that indicates how long it takes to complete 99% of requests within a system. It is valuable to also tune hyperparameters of ANN, whenever there is a change in the model. Authors refer [Google Vizier](https://cloud.google.com/vertex-ai/docs/vizier/overview) as a tool used for this optimization task.

### Meta experiments
Meta also utilized Faiss library as their ANN service (no wonders, it's developed by them). They share many learning so if you are interested in what to optimize when optimizing performance of ANN see [original paper](https://arxiv.org/abs/2006.11632). One tip that is also mentioned in Etsy paper is to tune ANN hyperparametrs when there is non-trivial model change.

Query embeddings are generated in real-time, while document embeddings are processed offline (with Spark) and integrated into the forward index. There is a query selection method implemented. These optimization help control the system's capacity usage, ensuring embedding-based retrieval is triggered only when it is likely to improve the search quality, balancing both speed and accuracy.

### Pinterest's serving architecture
PinnerFormer model is inferred daily. Only users who engaged on Pinterest in the past they have their embeddings updated. User embeddings are stored in key-value store for online serving. Architecture diagram is shown below.
<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Pinterest-serving-architecture.png" width="500" />
</div>
<p></p>
Having offline workflow, allows for using larger models, which in turn helps models learn more informative embeddings. They also utilize HNSW for storing item (Pin) embeddings. 


## Some cool ideas

### Human annotated data
Both JD and Mata used human annotated data. The data obtained with help of annotators was used to further improve the training, by correcting corner cases and incorporating prior knowledge. Some usefull training examples can be artificially created with domain knowledge. Some examples:
*  *cellphone cases* as generated as negative items for query *cellphone*. They share similar word "cellphone" but have different semantic meaning.
* *iPhone 16* items can be generated as positive items for a query *newest largest screen iphone*.

### Ensembling models
An interesting direction used in Meta was to enseble models trained with different "hardness" of negatives. They employ one model that focuses on recall - retreving similar items, and second one that focuses on precision - differenting between small set of retrieved items. You can think about "Retrieval" model as a model that was trained on random negatives. "Precision" model is trained on non-clicked but observed items from positive session. This enseble can be achieved as either weighted concatenation of models or a cascade model. Both of this approaches turns out to be effective. 

### Clustering layer to increase diversity
For Ebay, their final recommedations suffer from low diversity. To overcome this problem they clustered their product index into $$100\ 000$$ clusters, each with centroid $$c_i$$. They query the index to find $$M$$ nearest clusters and then in each cluster they look for $$m_ii$$ items (for more details see [original paper](https://arxiv.org/abs/2102.06156)). Results of *no-clustering* vs *clustering* can be seen on image below.

<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Ebay-clusters-diversity.png" width="600" />
</div>
What clustering essentialy does it creates "pseudo-catalog". Items with similar content are organise together in the same clusters. You can balance diversity and retrieval metrics with balancing $$M$$ parameter (with $$M=1$$ this idea degenerates to standard KNN). More about this in [this post]({% post_url 2024-10-01-personalized-retrieval-ebay %})! 

### ANN based product boosting

Main idea is to enrich product embeddings with additional numerical features (like shop popularity). This is done after model training. Both product and query embeddings are expaneded via conacatenation. New *hydrated vectors* are defined as follows:
$$p'=concat([p;f(p)])$$, $$q'=concat([q,w])$$ where $$p$$ and $$q$$ are product and query embeddings, respectively, $$f(p)$$ is a feature vector of numerical features, and $$w$$ is (learnable) constant vector with the same dimension as $$f(p)$$. For serving they index *hydrated vectors* instead of normal ones, final score is comupted as usual, with dot product between vectors $$p'$$ and $$q'$$.

One may wonder why not train these weights ($$w$$)during model training. This is actually really good question. Citing the authors of Etsy paper:

<div style="text-align:center;">
<i>However, compared to textual features like query or title text, static quality features like shop popularity are sensitive to negative sampling approaches and can easily over-fit on our proxy metrics without careful tuning and also do not optimize for recall directly.</i>
</div><p></p>

Therefore authors optimized them after training. They used black-box bayesian optizmiation ([skopt](https://scikit-optimize.github.io/stable/)) to learn query weights that optimize recall on items purchased after model trainign window. Query weights are general and do not depent on given query.




## Results
Long story short: <u>all models improve current systems</u>. This was tested with A/B tests with is much more reliable way of testing than offline ranking metrics. It doesn't mean these models completely push out existing solution - they show biggest improvement in ares like long tail queries and for that reason it's worth deploying them, but not necessarily for the whole traffic. 

For Etsy they reported improved CVR (conversion rate) by 2.63% and OSPR (organic search purchase rate) by 5.58%. Personalized model variants had a greater impact on the signed-in and habitual buyer segments. Sample recommendations are shown below.

<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Etsy-sample-recommendations.png" width="500" />
</div>
Adding more context definitely changes recommendations. At the sime time it does not completly overtake what is shown.


For JD, A/B tests took over 2 weeks and over 10% of the whole site's traffic. They are shown below.



<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/JD-ab-tests.png" width="400" />
</div>

There are only relative improvements shown here. Metrics are: user conversation rate (UCVR), gross merchandise value (GMV), and query rewrite rate (QRR). In all metrics we can see improvements, especially on long-tail queries. An interesting qeustion to ask is whether it makes sense to introduce user features - numbers from the table tell us the biggest difference in metrics is between baseline model and *1-head* version of their Two-Tower model. Relative difference between *2-head* and personalized version of the model (*1-head-p13n*) is rather miniscule, and for UCVR metric it scores even worse than *2-head*.

For Pinterest they are few interesting tables to discuss. But firstly, the data. Dataset used for evaluation consist of data from 2 weeks period, coming from users that were not persent in training data. Their item (Pin) index consists of 1M Pins.
<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Pinterest-online-vs-offline-recall.png" width="400" />
</div>
<p></p>
Authors compared two models with ranging inference frequency - inferring test set just one time, daily and in realtime. Moving from realtime to batch inference drops Recall@10 by 13.9% when training on SASRec objective, but only by 8.3% when using a dense all action objective (PinnerFormer). P@90 tells us what fraction of the index of 1M Pins accounts for 90% of the top 10 retrieved results over a set of users. It a diversity measure, that focuses on global diversity across all users. 
Big takeaway is that innferring in online is superior over inferring in online. Moreover *dense all action prediction* decreases model's sensitivty to short-term variations, and helps to learn more stable uer interests.
<div style="text-align:center;">
<i> There is still a nontrivial gap between realtime performance and daily inference performance, but given improvements over our baseline of PinnerSage, and the high cost and infra complexity of inferring PinnerFormer in realtime, we view this as an acceptable tradeoff. </i>
</div>
<p></p>
PinnerSage is their current production system for generating user embeddings. See orginal paper for detailed comparison of Pinnerformer and PinnerSage.
<div style="text-align: center;">
  <img src="/resources/general-personalized-retrieval/Pinterest-ab-tests-ads.png" width="350" />
</div><p></p>

Using user embeddings from Pinnerformer in downstream Ads models improves the results on many surfaces.

## Summary

This article explores how major companies like Meta, JD, Etsy, Ebay, and Pinterest implement personalized retrieval systems in their recommendation engines. The key findings include:

1. **Architecture Patterns**: Most companies employ variations of the Two-tower model architecture, which decouples query and item towers for efficient retrieval. This design allows for efficient Approximate Nearest Neighbors (ANN) search during inference.

2. **Feature Engineering**: Companies utilize diverse features including:
   - User demographics and social connections
   - Historical user behavior
   - Item metadata (titles, categories, etc.)
   - Location information
   - Temporal features

3. **Training Approaches**:
   - Loss functions: Max margin triplet loss and sampled softmax are commonly used
   - Negative sampling: Mixed Negative Sampling (combining random and hard negatives) proves effective
   - Human-annotated data helps improve model performance in corner cases

4. **Serving Optimizations**:
   - Offline processing of item embeddings
   - Real-time query embedding generation
   - Efficient ANN implementations (using libraries like Faiss)
   - Model sharding for handling multiple models
   - Trade-offs between inference frequency and performance

5. **Innovative Techniques**:
   - Multi-head query towers for handling polysemous queries
   - Clustering for improving recommendation diversity
   - ANN-based product boosting
   - Model ensembling for balancing recall and precision

6. **Results**: All implementations show significant improvements over baseline systems, particularly for long-tail queries. The improvements are validated through A/B testing, with metrics showing:
   - Increased conversion rates
   - Higher gross merchandise value
   - Better user engagement
   - Improved diversity in recommendations

The article demonstrates that while personalized retrieval systems can be complex, they provide substantial value in improving recommendation quality and user experience. The key is finding the right balance between model sophistication, serving efficiency, and business requirements.

## References

* [Personalized Embedding-based e-Commerce Recommendations at eBay](https://arxiv.org/abs/2102.06156)

* [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632)

* [Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning](https://arxiv.org/abs/2006.02282)

* [Unified Embedding Based Personalized Retrieval in Etsy Search](https://arxiv.org/abs/2306.04833)

* [PinnerFormer: Sequence Modeling for User Representation at Pinterest](https://arxiv.org/abs/2205.04507)

* [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)

* [Optimizing Dense Retrieval Model Training with Hard Negatives](https://arxiv.org/abs/2104.08051)
