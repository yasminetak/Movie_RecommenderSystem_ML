# --------Data preparation :------------ 
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data set 
ratings_data = pd.read_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/ml-latest-small/ratings.csv')
movies_data = pd.read_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/ml-latest-small/movies.csv')
print(" -Rating data : \n " , ratings_data.head())
print(" -Movies data : \n " , movies_data.head())
print(" -Ratings statistics : \n",ratings_data.describe())
# drop irrelevant columns like timestamp because it is not required to build the recommender system 
ratings_data.drop('timestamp', axis=1, inplace=True)
print(ratings_data.isnull().sum())
# drop rows with missing values
ratings_data.dropna(inplace=True)
# check for duplicates and remove em
print("numbers of duplications : ", ratings_data.duplicated().sum())
ratings_data.drop_duplicates(inplace=True)
# number of users and movies
unique_user = ratings_data.userId.nunique(dropna = True)
unique_movie = ratings_data.movieId.nunique(dropna = True)
print(" -Number of unique user:")
print(unique_user)
print(" -Number of unique movies:")
print(unique_movie)
# Merge the dataset, merge the ratings_data and movies_data dataframes
merged_data = pd.merge(ratings_data, movies_data, on='movieId')
# finalize the dataset : Drop any remaining irrelevant columns and rename columns if necessary
merged_data.rename(columns={'title': 'movie_title'}, inplace=True)
print(" -The Merged data :\n", merged_data.head())
# save the prepared data in new file prepared_data.csv
merged_data.to_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/prepared_data.csv', index=False)

# ------------Data Exploration :---------------
# load the prepared data set 
data = pd.read_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/prepared_data.csv')
print(" -The Prepared Data : \n", data.head())

# List of all genres
genres = []
for genre in movies_data.genres:
    x = genre.split('|')
    #print(x)
    for i in x:
        if(i not in genres):
            genres.append(str(i))
print(" -List of Genres : \n", genres)
# List of all Movie titles
titles = []
for movie_title in movies_data.title:
    if(movie_title not in titles):
        titles.append(str(movie_title))
print(" -List of Movie Titles : \n", titles)
# To see the totalRatingCount of movies 
combine_movie_rating = data.dropna(axis = 0, subset = ['movie_title'])
movie_ratingCount = (combine_movie_rating.
     groupby(by = ['movie_title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['movie_title', 'totalRatingCount']]
    )
print(" -Movie ratingCount :\n ",movie_ratingCount.head())
# Movies that received the highest number of ratings from User
highest_number_of_rating = data.groupby('movie_title')[['rating']].count()
print(" -Movies that received the Highest Rating from User : \n",highest_number_of_rating)
# Identify the most popular movies
most_popular = data.groupby(['movie_title']).size().sort_values(ascending=False)[:10]
print("Most popular movies : \n ", most_popular)
# basic statistics, mean, median for the data
print("Statistics of Data :\n",data.describe())

# PLOTS
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
# plot rating frequency of each movie(how many time a movie has been rated)
movie_freq = pd.DataFrame(ratings_data.groupby('movieId').size(),columns=['count'])
print(" -How many times a movie have been rated :\n",movie_freq.head())
# plot movie rating freq
movie_freq_copy = movie_freq.sort_values(by='count',ascending=False)
movie_freq_copy=movie_freq_copy.reset_index(drop=True)
plt.figure(figsize=(12, 8))
plt.title('Rating Frequency of Movies')
plt.xlabel('Number of Movies')
plt.ylabel('Rating Frequency of Movies')
plt.yscale('log')
plt.plot(movie_freq_copy['count'])
plt.show()


# ----------Model Implementation with knn: --------------
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# Load preprocessed dataset
data = pd.read_csv('C:/Users/dell/Desktop/Master2022-2023/S2/Machine Learning/Projet Recommender system/prepared_data.csv')
# Create a pivot table for item-based collaborative filtering
item_user_matrix = data.pivot_table(index='movieId', columns='userId', values='rating')
print("The matrix \n :", item_user_matrix.head())
# Replace missing values in the matrix with mean
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

# # Make recommendations for a user
# user = int(input("User id:"))
# user_index = user -1
user_index = 22 # choose a user ID
user_ratings = item_user_matrix.T.iloc[user_index, :].values.reshape(1, -1)
distances, indices = model_knn.kneighbors(user_ratings, n_neighbors=10)
# Print the top  recommended movies for the user
movie_ids = []
for i in range(len(indices.flatten())):
    if i == 0:
        # print('Recommendations for user {}:'.format(user_id))
        print('Recommendations for user {}:'.format(user_index))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, item_user_matrix.columns[indices.flatten()[i]], distances.flatten()[i]))    
#       movie_id = item_user_matrix.index[indices.flatten()[i]]
#       movie_ids.append(movie_id)
# # Get the movie titles for the recommended movies
# recommended_movies = data[data['movieId'].isin(movie_ids)][['movieId', 'movie_title']]
# recommended_movies = recommended_movies.drop_duplicates(subset='movieId')
# # # Print the recommended movies with their IDs and titles
# print(recommended_movies.to_string(index=False))










