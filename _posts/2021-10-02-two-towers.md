---
layout: post
title: "Two Towers to rule them all"
author: Mateusz Marzec
date: 2024-09-25
---


There are two encoders - one for queries and one for documents. Depending on a scenario they can be named differently. Former is often named as query/item/user encoder (or tower). Latter is most commonly named item/document encoder (or tower). You can think of this second tower as a tower that generates outputs - so if we want recommends items for a user we have a user and item towers.


Tower names and what do we input to them may change, but the most important parts of these architectures are:
* towers are fully disentangled
* we can input different types of features to each tower
* each tower compute a representation, which are then combined via similarity function to obtain 