---
layout: post
title:  "Recommender systems introduction"
date:   2023-08-14 07:44:17 +0100
author: Mateusz Marzec
---


In this tutorial let's focus our attention on recommender systems. These are complex but necessary elements for each business with abundant items and users. After this class I want you to be able to answer the following questions:
1. Why,
2. What, 
3. How

Of course all in the context of recommender systems.

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
Let us assume $$f$$ is the number of factors. Now we will try to model user-item interactions in $$\mathbb{R}^f$$ space. With each item $$i$$ and each user $$u$$ we associate vectors $$q_i \in \mathbb{R}^f$$ and $$p_u  \in \mathbb{R}^f$$, accordingly. Vector $$q_i$$ represents items embeddings and vector $$p_u$$ represents users embeddings. The values of $$q_i$$ measure the extent of which a given item has some factor. Similarly, the values of $$p_u$$ show how much interest a given user has for a specific item factor. For user $$u$$ the overall interest for item $$i$$ is then captured by a dot product $$q_i^T p_u$$. Assuming we have access to users ratings we can write a formula for rating prediction:

$$
\hat{r}_{ui} = q_i^T p_u.
$$

But this is explicit feedback case. We will focus on binary labels. In case of binary variable, we want to estimate probability of interaction, denoted $$p(u,i)$$. We can also use matrix factorization, with prediction formula as follows:

$$
p(u,i) = \sigma(q_i^T p_u),
$$

where $$\sigma$$ is logit function (same as in logistic regression). A visual example of how matrix factorization works is presented below:

<figure>
  <img
  src="/resources/recommender-systems-introduction/matrix-factorization-vis.png"/>
  <figcaption class="figure-caption text-center">How matrix factorization works. (<a href="https://developers.google.com/machine-learning/recommendation/collaborative/matrix?hl=en">Image source</a>).</figcaption>
</figure>
In this example we decompose user-item interaction matrix into two 2 dimensional metrices. One for user representation and one for item representation. To obtain score for a given user-item pair we just take the dot product between correct vectors.

TODO: add more equations, a bit more theory, 1-2 images

## Hands-on example
```python
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm
```

## Read the data
Read the data, then apply binarization. We convert the ratings to '1' when rating value is higher than 3. 


```python
import zipfile
import pandas as pd
with zipfile.ZipFile("./../data/ml-latest-small.zip") as z:
    with z.open("ml-latest-small/ratings.csv") as f:
        ratings = pd.read_csv(f, delimiter=",")
```


```python
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings['datetime'] = ratings['timestamp'].apply(lambda x: dt.datetime.fromtimestamp(x / 1000.0))
ratings["date"] = pd.to_datetime(ratings['datetime']).dt.date
interactions = ratings[ratings.rating > 3]
ratings.shape, interactions.shape
```




    ((100836, 6), (61716, 6))




```python
interactions.userId.nunique(), interactions.movieId.nunique()
```




    (609, 7363)



## Split the data
Split the data based on timestamp. First let's inspect number of interactions per day in our dataset.


```python
n_interactions = interactions.groupby("date").agg(n_obs=("userId","count")).reset_index()
n_interactions
plt.bar(n_interactions.date, n_interactions.n_obs)
plt.title("Number of interactions in given day")
plt.ylabel("n interactions")
plt.xticks(rotation=45)
plt.show()
```


    
![png](/resources/recommender-systems-introduction/movielens-countt-agg-daily.png)
    


Let's split the data on the last day. So in our example this will be day "1970-01-18". <br />
**Question: <br /> What are the advantages and disadvantages of such split compared to other types of splits? For other types of splitting consider random split* and leave-one-out split\*\*.** <br />
\* Random split is just randomly selecting interactions for test set. <br />
\*\* Leave-one-out split is selecting just one item for each user. Assume that we know temporal ordering and in such case leave-one-out selects last item from each user interaction history. Recall, that's not the same ase leave-one-out cross validation strategy!!!

**Answer:** 
Greatest advantage of this type of split is it's robustness to data leakages phenomena. For random split is trivial to see why leakage can happen. In leave-one-out split, last interaction of certain user can have a timestamp smaller than some observations from test set. This is dangerous, as some information from train data can leak to test set. One of the disadvantages is poor users coverage. Having test set taken from just one day, most of the users will have exactly zero interactions during that day (so we won't know how model will handle them in real scenario). Moreover, some users will have many interactions, other only one or two. Most of the ranking metrics at some point average the score over users - they don't consider that some users had more interacions in test set.


```python
split_date = "1970-01-18"
train = interactions[interactions.datetime < split_date]
test = interactions[interactions.datetime >= split_date]
train.datetime.max(), len(train), test.datetime.min(), len(test)
```




    (Timestamp('1970-01-17 23:59:44.741000'),
     51431,
     Timestamp('1970-01-18 00:04:28.014000'),
     10285)



Our train and test datasets look valid. We have around 50k and 10k observations in train and test, respectively. Due to the nature of choosen split latter number can change. <br />
**Question: <br />
Why is that?**

**Answer:** <br />
We haven't checked if all users (or items) from test set have at least one interaction in train set. If there is no interaction for given user / item in train set the model won't be able to learn how to generate recommendations for given user or item. Therefore, we may need to remove users (or items) which are not present in train set


```python
users, items = train.userId.unique(), train.movieId.unique()
n_users, n_items = len(users), len(items)
users_df = pd.DataFrame(users, columns=["userId"])
items_df = pd.DataFrame(items, columns=["movieId"])
u_to_ids = {user:idx for idx,user in enumerate(users)}
i_to_ids = {item:idx for idx,item in enumerate(items)}
train["users_mapped"] = train["userId"].apply(lambda x: u_to_ids[x])
train["items_mapped"] = train["movieId"].apply(lambda x: i_to_ids[x])
n_users, n_items
```
    (532, 6236)

```python
print(f"Test n obs before merge = {test.shape[0]}")
test_valid = test.merge(users_df, on="userId").merge(items_df, on="movieId")
print(f"Test n obs after merge = {test_valid.shape[0]}")
```

    Test n obs before merge = 10285
    Test n obs after merge = 803
    

It looks like we have removed more than 90% of test set! This is very rare situation, but we are working on educational dataset, so this is not that important. It is just a subset of the whole MovieLens dataset, and we don't know how exactly the data was collected.

## Model
In currect section we will implement Matrix Factorization model (and Neural Matrix Factorization model) 
### Matrix Factorization
**Exercise:**<br /> Implement matrix factorization model. Use the cell below as a guide (change given functions).


```python
class MF(nn.Module):
    def __init__(self, n_factors=None):
        pass
    
    def forward(self, user, item):
        pass
        
```


```python
class MatrixFactorization(nn.Module):
    def __init__(self, n_factors=None, n_users=None, n_items=None):
        super().__init__()
        self.n_factors = n_factors
        self.users = nn.Embedding(n_users, n_factors)
        self.items = nn.Embedding(n_items, n_factors)
    
    def forward(self, user, item):
        user_emb, item_emb = self.users(user), self.items(item)
        return (user_emb * item_emb).sum(dim=1)
```

### Neural Matrix Factorization


```python
# TODO
```

## Evaluation
Evaluation of the recommender system can be performed in various settings. The two main ways to do so are online evaluation and offline evaluation. The online evaluation uses a continuous stream of users feedback to access system performance. In practice, it is often difficult to design settings allowing for such evaluation. Offline evaluation is based on historical data and it is the most common way of evaluating recommender system models. When designing evaluation methodology for a recommender system, we have to consider our goals. In the early stages of research in this field, calculating RMSE, MSE or MAE (Mean Absolute Error) was a very popular way of accessing the performance of the recommender system. This was, of course, connected with the data availability - the main datasets available online consisted of explicit ranking. Therefore, recommendation problem was often treated as a regression problem so typical regression metrics were used. When the data was in a binary form - metrics based on classification were used. The current standard are ranking based metrics. They give the biggest weights to the items placed at the top of the ranking list, which mimics the goal of top-k recommendation. <br /> In this tutorial we will focus on **Recall@k** which is a metric build upon Recall used for standard classification problems. For a given user we can define: 

$$Recall@k = \frac{number\ of\ relevant\ items}{total\ number\ of\ items},$$ 

the number $$k$$ is the length of the recommendation list. To calculate Recall@k for the whole dataset we average the values of recall for each user. <br />

As we don't have negative interactions we can sample (again ;)) negatives for users from test set. For each user we sample 100 items, and consider them negative. Then if we want to calculate Recall@10 for user $$u$$ with two items in test set, we need to score all 102 items, and order them accordingly to obtained scores (higher score means higher probability of interaction). For Recall@10 we need to check how many of these two relevant items are in top-10 recommendation list and divide it by the total number of relevant items for this user (=2 in this case). 


**Exercise:**<br />
Implement function for calculating the recall for a given user. Then sample 100 negative items for each user (items have to be different from what is in given user's test set). Calculate recall at cutoffs k=5,10,30 for untrained model. Note: Your recall function doesn't have to have specific arguments defined below, but the solution will refer to them.


```python
def recall_k(k=10, annotated_preds=[]):
    pass
    # k - cutoff value
    # preds - list with tuples (probability, 0/1) indicating predicted scores and label
```

**Solution:**


```python
test_positives = test_valid.groupby("userId")["movieId"].agg(lambda x: x.tolist())

items_set = set(items)
positives_test = {}
negatives_test = {}
for u,i_pos in zip(test_positives.index, test_positives):
    sampled = np.random.choice(list(items_set - set(i_pos)), size=100, replace=False)
    positives_test[u] = i_pos
    negatives_test[u] = sampled
    
# encode users and items for model
positives_test_encoded = {}
negatives_test_encoded = {}
for u,itms in positives_test.items():
    positives_test_encoded[u_to_ids[u]] = [i_to_ids[i] for i in itms]
for u,itms in negatives_test.items():
    negatives_test_encoded[u_to_ids[u]] = [i_to_ids[i] for i in itms]
```


```python
# score positives and negatives
def score(model, positives_test_encoded, negatives_test_encoded):
    scored = defaultdict(list)
    for u, itms in positives_test_encoded.items():
        u_tensor = torch.tensor([u]).repeat(len(itms))
        itms = torch.tensor(itms)
        # evaluate model:
        model.eval()
        with torch.no_grad():
            preds = model(u_tensor, itms)
        for p in preds:
            scored[u].append((p.item(),1))

    for u, itms in negatives_test_encoded.items():
        u_tensor = torch.tensor([u]).repeat(len(itms))
        itms = torch.tensor(itms)
        # evaluate model:
        model.eval()
        with torch.no_grad():
            preds = model(u_tensor, itms)
        for p in preds:
            scored[u].append((p.item(),0))
    return scored
```


```python
def recall_k(k=10, annotated_preds=[]):
    # k - cutoff value
    # preds - list with tuples (probability, 0/1) indicating predicted scores and label
    n = min(k, sum([i[1] for i in annotated_preds]))
    s = sorted(annotated_preds, key=lambda x: x[0], reverse=True)
    r = sum([i[1] for i in s[:k]])
    return r / n
```


```python
# init random model
model = MatrixFactorization(n_factors=20, n_users=n_users, n_items=n_items)
scored = score(model, positives_test_encoded, negatives_test_encoded)
for k in [5,10, 30]:
    recalls = [recall_k(k, pred) for pred in scored.values()]
    mean_recall = np.mean(recalls)
    print(f"Recall@{k} = {mean_recall}")
```

    Recall@5 = 0.18461538461538465
    Recall@10 = 0.18119658119658125
    Recall@30 = 0.2642334231890363
    


```python
# helper function, we will use it later during trainig
def score_recall(model, positives_test_encoded, negatives_test_encoded, k):
    scored = score(model, positives_test_encoded, negatives_test_encoded)
    recalls = [recall_k(k, pred) for pred in scored.values()]
    return np.mean(recalls)
```

## Training
Now, we will train implemented models. We need to define some training hyperparameters (learning rate, batch size and the number of epochs), initilize optimizer and model and implement custom dataset class (we could live without it, but it simplifies code during training) and training loop.


```python
lr = 1e-3
batch_size = 64
n_epochs = 40
```

For the loss function we will use Binary cross entropy, as we are treating our problem as a classification problem.


```python
model = MatrixFactorization(n_factors=20, n_users=n_users, n_items=n_items)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.BCEWithLogitsLoss()
```


```python
class MovieLensDataset(Dataset):
    def __init__(self, data, output_col="output"):
        self.users = data.users_mapped.values
        self.items = data.items_mapped.values
        if output_col:
            self.output = data[output_col].values
        else:
            self.output = np.ones(self.users.shape)
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        return (self.users[idx], self.items[idx], self.output[idx])
```


```python
dataset = MovieLensDataset(train, output_col=None)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### Train without sampling
In classification problems we have both negative and positive instances, but here we only have positives. In this section we will try to train a model only on positive interactions.


```python
def train_fn(model, optimizer, n_epochs, dataloader, verbose_n_steps=None, eval_recall=True, k=10):
    epoch_losses, recalls = [], []
    global_step = 0
    epoch_loss = []
    for epoch in range(1, n_epochs+1):
        print(f"Running epoch {epoch}")
        tmp_loss = []
        for user, item, output in tqdm(dataloader):
            optimizer.zero_grad()

            # Predict and calculate loss
            prediction = model(user, item)
            loss = loss_function(prediction, output)

            # Backpropagate
            loss.backward()

            # Update the parameters
            optimizer.step()

            global_step += 1
            epoch_loss.append(loss.item())
            
            if verbose_n_steps:
                # store current loss
                tmp_loss.append(loss.item())
            if verbose_n_steps and global_step % verbose_n_steps == 0:
                avg = np.mean(tmp_loss)
                print(f"Step = {global_step}, moving average loss = {avg}")
                tmp_loss = []
                
        e_loss = np.mean(epoch_loss)
        epoch_losses.append(e_loss)
        print(f"Avg loss after epoch {epoch}, is equal to {e_loss}")
        
        if eval_recall:
            r = score_recall(model, positives_test_encoded, negatives_test_encoded, k)
            print(f"Recall@{k} (test set) after epoch {epoch}, is equal to {r}")
            recalls.append(r)
            
    return epoch_losses, recalls
```


```python
losses, recalls = train_fn(model, optimizer, n_epochs, dataloader, verbose_n_steps=None)
```

    Running epoch 1
    100%|███████████████████████████████████████████████████████████████████████████████| 804/804 [00:02<00:00, 397.44it/s]
    Avg loss after epoch 1, is equal to 1.8510106407988793
    Recall@10 (test set) after epoch 1, is equal to 0.24658119658119657
    
    Running epoch 2
    100%|███████████████████████████████████████████████████████████████████████████████| 804/804 [00:02<00:00, 308.31it/s]
    Avg loss after epoch 2, is equal to 1.7544759394481668
    Recall@10 (test set) after epoch 2, is equal to 0.2388888888888889
    
    .
    .
    .
    
    Running epoch 40
    100%|███████████████████████████████████████████████████████████████████████████████| 804/804 [00:02<00:00, 292.07it/s]
    Avg loss after epoch 40, is equal to 0.39081827431721494
    Recall@10 (test set) after epoch 40, is equal to 0.35042735042735046

```python
# plot training loss
def plot_vector(vector, title=None, xlabel="epoch", ylabel="loss", figsize=(12,5)):
    plt.figure(figsize=figsize)
    plt.plot([i+1 for i in range(len(vector))],  vector, "o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([i+1 for i in range(len(vector))])
    plt.show()

plot_vector(losses, ylabel="loss", figsize=(14,5))
```


    
![png](/resources/recommender-systems-introduction/mf_train_loss.png)
    



```python
# plot recall@10 on test set
plot_vector(recalls, ylabel="Recall@10", figsize=(14,5))
```


    
![png](/resources/recommender-systems-introduction/mf_recall10_test.png)
    



```python
scored = score(model, positives_test_encoded, negatives_test_encoded)
for k in [5,10, 30]:
    recalls = [recall_k(k, pred) for pred in scored.values()]
    mean_recall = np.mean(recalls)
    print(f"Recall@{k} = {mean_recall}")
```

    Recall@5 = 0.31538461538461543
    Recall@10 = 0.35042735042735046
    Recall@30 = 0.48000679156622084
    

### Train with sampling
Here, we want to enrich training data with negative instances. There are many ways of sampling, but for simplicity we will focus on simplest approach - random sampling. It simply samples observations uniformly from (user, item) pairs. <br />

**Exercise:** <br />
For each interaction from train set sample one negative interaction with random sampling. Merge that with training data and train model with the same hyperparameters, for the same number of epochs and compare obtained results. <br />

**Solution:**


```python
n_negatives = train.shape[0]
users_sampled = np.random.choice(users, n_negatives)
items_sampled = np.random.choice(items, n_negatives)
negative_df = pd.DataFrame({"userId":users_sampled, "movieId":items_sampled, "output":np.zeros(n_negatives)})
```


```python
train_sampled = pd.concat([train[["userId", "movieId"]], negative_df])
train_sampled = train_sampled.fillna(1)
train_sampled["users_mapped"] = train_sampled["userId"].apply(lambda x: u_to_ids[x])
train_sampled["items_mapped"] = train_sampled["movieId"].apply(lambda x: i_to_ids[x])
```


```python
dataset_sampled = MovieLensDataset(train_sampled)
dataloader_sampled = DataLoader(dataset_sampled, batch_size=batch_size, shuffle=True)


model_negatives = MatrixFactorization(n_factors=20, n_users=n_users, n_items=n_items)

optimizer = torch.optim.Adam(model_negatives.parameters(), lr=lr)
loss_function = nn.BCEWithLogitsLoss()
```


```python
losses, recalls = train_fn(model_negatives, optimizer, n_epochs, dataloader=dataloader_sampled)
```

    Running epoch 1
    100%|█████████████████████████████████████████████████████████████████████████████| 1608/1608 [00:03<00:00, 408.64it/s]
    Avg loss after epoch 1, is equal to 1.8333566987359216
    Recall@10 (test set) after epoch 1, is equal to 0.17735042735042736
    
    Running epoch 2
    100%|█████████████████████████████████████████████████████████████████████████████| 1608/1608 [00:03<00:00, 418.38it/s]
    Avg loss after epoch 2, is equal to 1.7072859640175888
    Recall@10 (test set) after epoch 2, is equal to 0.1735042735042735
    
    .
    .
    .

    Running epoch 40
    100%|█████████████████████████████████████████████████████████████████████████████| 1608/1608 [00:03<00:00, 504.34it/s]
    Avg loss after epoch 40, is equal to 0.42990228001413516
    Recall@10 (test set) after epoch 40, is equal to 0.2811965811965812

```python
# plot loss on train set
plot_vector(losses, ylabel="loss", figsize=(14,5))
```


    
![png](/resources/recommender-systems-introduction/mf_negatives_train_loss.png)
    



```python
# plot recall@10 on test set
plot_vector(recalls, ylabel="Recall@10", figsize=(14,5))
```


    
![png](/resources/recommender-systems-introduction/mf_negatives_recall10_test.png)
    



```python
scored = score(model_negatives, positives_test_encoded, negatives_test_encoded)
for k in [5,10, 30]:
    recalls = [recall_k(k, pred) for pred in scored.values()]
    mean_recall = np.mean(recalls)
    print(f"Recall@{k} = {mean_recall}")
```

    Recall@5 = 0.2923076923076923
    Recall@10 = 0.2811965811965812
    Recall@30 = 0.42681061824536726

 Results are quite suprising. It looks like adding negatives does not improve model performance. There is of course more to it. Random sampling is simplest and weakest sampling methods. Using more advanced sampling techniques (like popularity sampling, where we sample items according to their popularity) could lead to different results. However, from our experiment we can conclude that adding negatives wasn't benefitial.   

## Conclusions
In this tutorial we have learned the basics of recommender system. We started with general use cases of recommender systems. We then defined what recommender system is and how one can be implemented. We outlined the basics of the most popular model - matrix factorization. Many more advanced models build on that foundation, so it's crucial to understand the nitty gritty of matrix factorization. We then moved to practical example. Our dataset was subset of MovieLens dataset. We performed exploratory data analysis and prepared data for modelling. Some tricks were discussed along the way. Lastly we compared influence of sampling negatives for choosen accuracy metric. 

