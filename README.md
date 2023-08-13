# Recommender Systems - introduction
In this class let's focus our attention on recommender systems. These are complex but necessary elements for each business with abundant items and users. After this class I want you to be able to answer the following questions:
1. Why,
2. What, 
3. How

Of course all in the context of recommender systems. Ehh... 

## Why
Recommender systems aim at providing personalized product suggestions to customers. In the era of gargantuan numbers of different products choosing the right one for you can be an arduous task [paradox of choice](https://en.wikipedia.org/wiki/The_Paradox_of_Choice). Recommender systems aim to solve this problem. Having access to user browse history, demographical data, items characteristics it aims at reducing the space of items to only a few relevant (usually 10,20,...). Big retailers often rely on them. On Amazon 35% of their products are bought from recommendations produced by their recommender systems. For Netflix this number is even more substantial [more here](https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers).
 
## What
Product recommendation is a complex problem. Preparing a list of products is just one of the steps. Website design, how each product will be presented, how many products to show, etc. all play vital role in succesful product recommendations. This tutorial will only consider how to prepare a personalized recommendation list. 
### Data
We may have different types of data to work with. The most basic datatype are user-item interactions. They are cornerstone for every recommender dataset. They encode the relationship between user and item. This relationship can be evaluated on Likert scale (ratings, like 1-5 stars) or by a binary variable. With each interaction we can associate additional user and item features. For instance in news recommendations we should consider the news title as an additional feature. From user side we can use users demographical data (gender, age, if such data is available). On of the often used categorisaction is **implicit feedback** and **explicit feedback** disctinction. 
**Implicit feedback** is a scenario when interactions are scored via binary variable. An example of such system is online shop with user browsing items. Each click can be considered '1' and each impression a '0'.
**Explicit feedback** is a scenario when interactions are scored on the ordered scale - like ratings. An example of such system is a website with movies. Each user can rate and comment on a movie. Each rating can be then treated as a real number. 
In practice, we more often have implicit feedback data. Not always the impressions will be available (so we may have data only about positive interactions), but this issue is tackled by negative sampling which will be discussed later on.

### Taxonomy
Considering different data types that we can input into the recommender system and various conceptual goals. We can classify recommender systems into one of the few categories. Their characteristics are listed in table below. 
![image](https://github.com/mefor44/mefor44.github.io/assets/61019250/385c759d-6692-46b8-aa15-bc348f752a7a)
The Collaborative filtering relies only on previous user behaviours, in contrast to the two former methods, which need some sort of user or item profile description. An example of such data is a sequence of product ratings given by a specific user. In the Collaborative filtering, we try to understand the relationships between users and items to identify new user-item connections. To learn the preferences of specific user u we do not limit ourselves to user u data (his ratings, or interactions with items) but we incorporate the data about other users behaviours. Doing so we can create latent representations for users and items (latent factor models) or find the "neighbors" - user or items with similar characteristics (neighborhood models). Latent factor models create low-dimensional representation for both users and items. Generally, factors are not interpretable. The most popular examples of latent factor models are matrix factorization techniques. These methods can provide both good scalability and predictive accuracy.
Above-mentioned neighborhood models rely on item-item or user-user similarities. They estimate
an unknown rating as a weighted average of similar items to the one that we want to rate. A more
detailed description of these approaches will be introduced in section "How".

## How
In this tutorial, we will focus on Collaborative filtering techniques.
### Problem definition

### Matrix factorization techniques
Let us assume $f$ is the number of factors. Now we will try to model user-item interactions in $Rf$ space. With each item i and each user u we associate vectors qi ∈ Rf and pu ∈ Rf , accordingly. Vector qi represents items embeddings and vector pu represents users embeddings. The values of qi measure the extent of which a given item has some factor. Similarly, the values of pu show how much interest a given user has for a specific item factor. For user u the overall interest for item i is then captured by a dot product qT i pu. Assuming we have access to users ratings we can write a formula for rating prediction
### Neighborhood methods
The main idea that stands behind Neighborhood models is fairly simple. We try to suggest
items that are liked by similar users or items which are similar to items liked by a specific user.
We can highlight two different approaches - one focused on users and one focused on items

## Hands-on example
```python
import pandas as pd
```


```python
!dir "./../data/"
```

     Volume in drive D is Nowy
     Volume Serial Number is 7EA6-BDA1
    
     Directory of D:\MM\mefor44.github.io\data
    
    12.08.2023  16:39    <DIR>          .
    12.08.2023  16:39    <DIR>          ..
    12.08.2023  16:39           978˙202 ml-latest-small.zip
                   1 File(s)        978˙202 bytes
                   2 Dir(s)  623˙882˙334˙208 bytes free
    


```python
df = pd.read_csv('./../data/ml-latest-small.zip', compression='zip', header=0, sep=',', quotechar='"')
```


```python
import zipfile
import pandas as pd
with zipfile.ZipFile("./../data/ml-latest-small.zip") as z:
    with z.open("ml-latest-small/ratings.csv") as f:
        ratings = pd.read_csv(f, delimiter=",")
```


```python
print(ratings.head())    # print the first 5 rows
```

       userId  movieId  rating  timestamp
    0       1        1     4.0  964982703
    1       1        3     4.0  964981247
    2       1        6     4.0  964982224
    3       1       47     5.0  964983815
    4       1       50     5.0  964982931
    


Sources:

