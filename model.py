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
# histogram of rating distribution of all movies
plt.hist(data['rating'])
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Movies Ratings')
plt.show()
# Identify the most popular movies
most_popular = data.groupby(['movie_title']).size().sort_values(ascending=False)[:10]
print("Most popular movies : \n ", most_popular)
# distribution of genres
genre_counts = data['genres'].value_counts()
print("genres : \n ", genre_counts)

# ----------Model Implementation : --------------
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
# split the dataset in training and testing set
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print("traindata\n", train_data.columns)
# create item-user matrix
item_user_matrix = train_data.pivot_table(index='movie_title', columns='userId', values='rating')
print("item-user matrice : \n", item_user_matrix)





