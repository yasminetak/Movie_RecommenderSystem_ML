# --------Data preparation :------------ 
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# load the data set 
ratings_data = pd.read_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/ml-latest-small/ratings.csv')
movies_data = pd.read_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/ml-latest-small/movies.csv')
# explore the data 
print(" -Rating data : \n " , ratings_data.head())
print(" -Movies data : \n " , movies_data.head())
# drop irrelevant columns like timestamp because it is not required to build the recommender system 
ratings_data.drop('timestamp', axis=1, inplace=True)
print(ratings_data.isnull().sum())
# drop rows with missing values
ratings_data.dropna(inplace=True)
# check for duplicates and remove em
print("numbers of duplications : ", ratings_data.duplicated().sum())
ratings_data.drop_duplicates(inplace=True)
# merge the dataset, merge the ratings_data and movies_data dataframes
merged_data = pd.merge(ratings_data, movies_data, on='movieId')
print("merged data : \n", merged_data.head())
# finalize the dataset : Drop any remaining irrelevant columns and rename columns if necessary
merged_data.rename(columns={'title': 'movie_title'}, inplace=True)
print("new merged data \n", merged_data)
# save the prepared data in new file
merged_data.to_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/prepared_data.csv', index=False)

# ------------Data Exploration :---------------
# load the prepared data set 
data = pd.read_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/prepared_data.csv')
print("Data : \n", data.head())
# basic statistics, mean, median
print(data.describe())
# plot number of ratings per movie
plt.figure(figsize=(12,6))
data.groupby('movie_title')['rating'].count().sort_values(ascending=False).head(25).plot(kind='bar')
plt.title('Number of Ratings per Movie')
plt.xlabel('Movie Title')
plt.ylabel('Number of Ratings')
plt.show()
# plot of rating distribution of all movies
plt.hist(data['rating'])
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Movies Ratings')
plt.show()
# Identify the most popular movies
most_popular = data.groupby(['movie_title']).size().sort_values(ascending=False)[:10]
print("Most popular movies : \n ", most_popular)
# distribution of genres
genre_counts = data['genres'].value_counts()
print("genres : \n ", genre_counts)
# Sort the DataFrame by ratings in descending order
sorted_movies = data.sort_values(by='rating', ascending=False)
# Print the sorted DataFrame
print("sorted movies by ratings : \n ",sorted_movies)

# ----------Model Implementation with knn: --------------
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
# Load preprocessed dataset
data = pd.read_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/prepared_data.csv')
print("Data : \n", data.head())
# Create a pivot table for item-based collaborative filtering
item_user_matrix = data.pivot_table(index='movieId', columns='userId', values='rating')
# Replace missing values with mean
item_user_matrix = item_user_matrix.apply(lambda x: x.fillna(x.mean()), axis=0)
# Split the data into training and testing sets
train_data, test_data = train_test_split(item_user_matrix.T.values, test_size=0.2)
print("traindata\n : ", train_data)
print("testndata\n : ", test_data)
# Train the model using k nearest neighbors
# Train the k-NN model with k=10
k = 10
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k)
model_knn.fit(train_data)
# Make recommendations for a user
user_id = 4 # choose a user ID
user_ratings = item_user_matrix.loc[:, user_id].values.reshape(1, -1)
distances, indices = model_knn.kneighbors(user_ratings, n_neighbors=10)
# Print the top 10 recommended movies for the user
movie_ids = []
for i in range(len(indices.flatten())):
    if i == 0:
        print('Recommendations for user {}:'.format(user_id))
    else:
        movie_id = item_user_matrix.index[indices.flatten()[i]]
        movie_ids.append(movie_id)
# Get the movie titles for the recommended movies
recommended_movies = data[data['movieId'].isin(movie_ids)][['movieId', 'movie_title']]
recommended_movies = recommended_movies.drop_duplicates(subset='movieId')
# Print the recommended movies with their IDs and titles
print(recommended_movies.to_string(index=False))












