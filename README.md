Music Recommendation System - Final Project : 

Group members :  Ariel Hedvat, Eitan Bakirov, Yuval Bakirov , Shiraz Israeli.

Background:  A recommendation system is a tool that uses data to understand user preferences and behaviours, suggesting songs and artists that match individual tastes. 
Our project is ‘music recommendation system’ It enhances music discovery, saving users time and effort while introducing them to new music they're likely to enjoy. During the project, we explored recommendation systems approaches in general and in music specifically, looking for a dataset to work with. Our goal was to build a system that suggests the top 10 songs to a user. We tried several models, evaluated and compared them, aiming to create an effective recommendation system.

Project description and explanation:
Our Data The Million Song Dataset, comprises two files: 
The first file includes: song ID, title, release (album), artist name, and release year.  
The second file contains user IDs, song IDs, and the corresponding play counts by users.

First step of the project was to separate explorations of both datasets to understand how to merge them effectively. Merging was done based on the user-song identification key to create a combined dataset for EDA, preprocessing, and model training.

Exploratory Data Analysis (EDA): we tried to understand the dataset through a deep analysis, focusing on songs, albums, artists, and users. It provided key insights for improved preprocessing and modelling.

Pre Processing: Based on what we found during the EDA step, we modified our combined dataset, dealing with the listen_count feature - generating a rating feature (explanation below) , dealing with the ‘year’ feature that needed a modification as the data continued year=0, reducing the number of users (who listen to less than 40 songs) and songs (heard by less than 200 users) according to insights we gained during the analysis phase of the data.

We generated a rating scale, providing a rating that represents how much a user likes a song, similar to star ratings for movies or restaurants. This involves dividing listen counts into ranges based on a threshold value, which is calculated using the mean and standard deviation of listen counts for a user. This approach allows for adjusting the sensitivity of the threshold to the variability of listen counts.

Our approach is performing one model together - the Rank-Based Popularity model, which differs from the others. Each pair in our group then performs different algorithms separately. We compared these models to determine the best one for our mission.

Rank-Based (POPULARITY) 
This model recommends songs based on their overall popularity among users, disregarding individual preferences. It ranks songs by average ranking to determine the top recommendations, making it simple and not requiring train-test splitting or complex training.

REST OF THE MODELS WORKFLOW : For the other models, we follow a consistent workflow, splitting the data into train and test sets (20% of rated songs per user were randomly selected as testing data, and the others were used as training data). We explained each model, used grid search cross-validation to find optimal hyperparameters, train it on the train set, and evaluate it on the test set using precision_recall_at_k metrics. Additionally, we tried to analyse each model based on its results and compared recommendations from each model for a random user to demonstrate their results and advantages.
We use the Surprise library (for most of them), which deals with explicit ranking data, to build our models. And we set useful functions for evaluating and implementing a recommendation system that we will use during the modelling. 
We accompanied each model with both mathematical and descriptive explanations, aiming to clarify their functioning and provide intuitive insights into their operations.
The models we used are:

COLLABORATIVE FILTERING (CF)
CF Recommends songs by analysing user preferences for similarities. The Similarity-Based approach, a type of CF, identifies users or items with similar listening patterns. However, as datasets grow, finding reliable similarities becomes challenging, leading to scalability issues and difficulties in recommending songs to new users or items with few interactions.
                              
Two common strategies for similarity based:
 
User-User Similarity-Based Model: A User-User Similarity-Based Model calculates recommendations by comparing a user's preferences with those of similar users. We used the KNN Basic model from the surprise library that measures similarity between users based on their rating patterns and recommends songs favoured by users with similar preferences. During the work we tried to show interesting interactions between the users and how this affects the prediction results of the model, based on the similarity matrix of the users.
Here are the model results: 
{'MAE': 0.5752644062349389,  'RMSE': 0.7805737501824392, 'Precision@10': 0.922,
 'Recall@10': 0.794, 'F_1 score': 0.853}. 

Item-item Similarity-Based Model:  In our Item-Item Similarity-Based Model, songs are recommended to users based on their similarity to items (songs) previously liked by the user. We utilised the KNN Basic model from the surprise library, which computes item-item similarity by analysing their co-occurrence patterns in user ratings. Additionally, we visualised and learned from the similarity matrix between items to provide insights into their relationships. 
Here are the model results: {'MAE': 0.5001130432743669,  'RMSE': 0.7308742395433399, 'Precision@10': 0.923, 'Recall@10': 0.782, 'F_1 score': 0.847}.

From analysing the similarity model results, we found that the similarity matrices for user-user and item-item , but unfortunately it didn't yield as interesting results as expected. The similarity scores between users were generally low, indicating either a lack of similar users in the dataset or a weak model that captures few connections between users. Despite expecting overlap in recommended songs for similar users, we found that the recommended songs were mostly different, except for one song. That could be due to the fact that the model's performance may be limited due to the dataset's lack of similar users.


Another CF approaches are matrix factorization and cluster-based:

Matrix Factorization (MF) is a Collaborative Filtering model that recommends songs based on similarities in users' preferences. It decomposes the user-song interaction matrix using SVD (Singular Value Decomposition), mapping users and items into a lower-dimensional latent space to predict ratings for new items and make personalised recommendations.
Using the  SVD model (by the surprise package) : After testing the model, here is the result of evaluation for the precision_recall_at_k function for the SVD  model :  
{'MAE': 0.5368864737029896,  'RMSE': 0.7444214574444593, 'Precision@10': 0.915,
 'Recall@10': 0.784, 'F_1 score': 0.844}

Our analysis of the SVD model showed that each latent factor captures unique aspects of user preferences. We found an interesting way of grouping users based on these factors, leading us to experiment with recommending songs to users in the same cluster. This approach involves calculating cosine similarity for user factors and using K-means clustering to group users with similar preferences.We provided an example of this approach for a random user,  identified the cluster of a random user, found all users in the same cluster, and recommended popular songs in the cluster that the random user had not yet listened to.
Then, we compare the recommendations that were suggested for this random user based on both approaches, and try to compare between them. 

MF EMBEDDINGS: We'll also explore the MF model using embedding layers to understand its mechanics better. Each user and song is represented in a latent space, capturing essential characteristics. Using input and embedding layers, we map IDs to dense vectors, then combine embeddings with a dot product operation to measure similarity. The result of the MF model with embedding (without the use of surprise): {Mean Absolute Error: 0.53121654805058 , Root Mean Squared Error (RMSE): 0.7288460386464208 }


CLUSTER BASED MODEL
We use here an implementation of a collaborative filtering method model named coClustering.
CoClustering, which falls under the category of Cluster-Based Collaborative Filtering, leverages the inherent structure within user-item interaction data to simultaneously cluster users and items, thus capturing underlying structures and similarities.

CoClustering seeks similarities among users and items within clusters, enriching insights through three dimensions:
Similarity among users: Users in the same cluster shared preferences, aiding collaborative filtering.
Similarity among items: Items in the same cluster have common attributes, facilitating recommendations.
Co-cluster behavior: Reflects joint user-item interactions, guiding personalised recommendations.

We tried to get a better understanding of how the model works and get a look at the clusters and the CoCluster. It seems that our attempts were indeed successful and it can be seen that similar users do receive similar recommendations of songs.

{'MAE': 0.5505471717748933,  'RMSE': 0.7564091327396436,  'Precision@10': 0.917,  'Recall@10': 0.778,  'F_1 score': 0.842}



CONTENT BASED MODEL
In this model , song features such as title, artist name, and release (Album) are used to find similar songs, without considering user listen count data or the rating we generated. This approach aims to recommend songs based on their textual features, not on how often users listen to them. It is a different approach from the other models, but we found it interesting to see this method and how it can recommend songs based on the textual features. 

By using TF-IDF with Cosine-Similarity we'll get an idea of how much the songs are similar to each other.
We will use TF-IDF to weight the frequency and importance of the different words in the 'title', 'artist name', and 'release' features which were merged together into one feature ‘text’.
Also, we use Cosine Similarity Score which is calculate an angle between the vectors of two songs to understand how similar they are (The vectors consist of the tf-idf scores of each word in the song metadata ['title', 'artist_name' and 'release']).
The larger the angle, the more the songs are similar.
Finally, we recommend a user a ‘K’ most recommended songs based on the songs he listened to.

We tried to evaluate the model by estimating how many of the songs that were recommended to the user had already been listened to by him before and he liked them (rating higher than 3). Unfortunately, we ran into runtime issues and had no time to finish properly :(





Comparing the models : 

 
Surprisingly it seems like all the models generated similar results with slight differences.
It seems that all the models show similar results and pretty much the same performance. It can be seen from the graph that the model with the lowest rmse performance is item-item, but in addition the MF embedding model brings rmse results of 0.72 (it does not appear in the displayed graph).


CONCLUSIONS : 

In summary, there are numerous ways to enhance the models and create a more effective recommendation system. However, within the given time frame, these are the accomplishments we managed to achieve.
# Music-Recommendation-System-Methods-
A Study about the process and methods used in music recommendation systems 
