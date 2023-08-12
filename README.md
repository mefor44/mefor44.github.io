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

### Taxonomy
Bla bla 
\begin{table}[H]
\resizebox{\linewidth}{!}{\begin{tabular}{|l|l|l|}
\hline
Approach                                      & Conceptual Goal                                                                                                                                                                       & Input                                                                                               \\ \hline
\multicolumn{1}{|l|}{Collaborative filtering} & \multicolumn{1}{l|}{\begin{tabular}[c]{@{}l@{}}Give me recommendations based on a collaborative approach, \\ that leverages the ratings and actions of my peers/myself.\end{tabular}} & \multicolumn{1}{l|}{\begin{tabular}[c]{@{}l@{}}User ratings,\\ community ratings\end{tabular}}     \\ \hline
\multicolumn{1}{|l|}{Content-based}           & \multicolumn{1}{l|}{\begin{tabular}[c]{@{}l@{}}Give me recommendations based on the content (attributes)\\ I have favored in my past ratings and actions.\end{tabular}}               & \multicolumn{1}{l|}{\begin{tabular}[c]{@{}l@{}}User ratings,\\ item attributes\end{tabular}}       \\ \hline
Knowledge-based                               & \begin{tabular}[c]{@{}l@{}}Give me recommendations based on my explicit specification\\ of the content (attributes) I want.\end{tabular}                                      & \begin{tabular}[c]{@{}l@{}}User specification,\\ item attributes,\\ domain knowledge\end{tabular} \\ \hline
\end{tabular}}
\caption{Different types of recommender systems, source: \cite{aggarwal}.}
\label{tab:rec-sys-taxonomy}
\end{table}

## How

### Problem definition



Sources:

