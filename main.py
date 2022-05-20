import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Pour montrer toutes les colonnes d'un dataframe
pd.set_option('display.max_columns', None)

# Chargement des données
data = pd.read_csv("/MarcoChavez98/Python_Project_MBFA/blob/main/tmdb_movies_data.csv")

# Création du dataframe
movies_df = pd.DataFrame(data)

# On enlève les colonnes que l'on ne souhaite pas exploiter
movies_df_trim = movies_df.drop(
    columns=['imdb_id', 'popularity', 'homepage', 'release_date', 'budget_adj', 'revenue_adj'])

# Enlever les films en doublon
# movies_df_trim["original_title"].drop_duplicates

# On crée un dataframe qui sera utilisé pour les content based recommendations
movies_data = movies_df_trim.loc[:, ["original_title", "cast", "director", "genres", "keywords", "overview"]]

# On change le nom du titre pour plus de clareté
movies_data.rename(columns={"original_title": "title"}, inplace=True)

# On regarde combien il y a de valeurs nulles
#print(movies_data.isna().sum())

# On remplace les valeurs nulles par des blanks
movies_data.fillna(" ", inplace=True)
# print(movies_data.isna().sum())


# On remplace les "|" par des virgules pour les colonnes concernées
movies_data["cast"] = movies_data["cast"].apply(lambda x: x.replace("|", ", "))
movies_data["genres"] = movies_data["genres"].apply(lambda x: x.replace("|", ", "))
movies_data["keywords"] = movies_data["keywords"].apply(lambda x: x.replace("|", ", "))
movies_data["director"] = movies_data["director"].apply(lambda x: x.replace("|", ", "))

# On combine genre+cast et tagline+keywords+overview
movies_data["combined"] = movies_data["genres"] + ", " + movies_data["cast"]
# movies_data["combined2"] = movies_data["tagline"] + " " + movies_data["keywords"] + " " + movies_data["overview"]

# movies_data["title"] = movies_data["title"].str.lower()
print(movies_data.head())
print(movies_data.info())
print(movies_data.shape)

# On extrait les valeurs uniques pour title, cast, director, genres, keywords
unique_titles = movies_data.title.unique()
actors = movies_data['cast'].str.split(', ', expand=True).stack()
unique_actors = actors.unique()
directors = movies_data['director'].str.split(', ', expand = True).stack()
unique_directors = directors.unique()
genres = movies_data['genres'].str.split(', ', expand = True). stack()
unique_genres = genres.unique()
keywords = movies_data['keywords'].str.split(', ', expand = True).stack()
unique_keywords = keywords.unique()
print(unique_keywords[:25])
print(len(unique_keywords))

#Streamlit
st.title('Movie recommendator :popcorn: :movie_camera:')
st.header('Spending more time choosing a movie than watching it?')
st.subheader('Try our movie recommendator based on your preferences!')
st.caption('Choose as many as you want')

title_choice = st.multiselect("Choose your favorite movies!", unique_titles)
actor_choice = st.multiselect("Choose your favorite actresses and actors!", unique_actors)
genre_choise = st.multiselect("Choose your favorite genres!", unique_genres)
keywords_choice = st.multiselect("Any keywords?", unique_keywords)


# Datacamp
#from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate the vectorizer object and transform the plot column
#vectorizer = TfidfVectorizer(max_df=0.7, min_df=10)
#vectorized_data = vectorizer.fit_transform(movies_data['combined'])

# Create Dataframe from TF-IDFarray
#tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names())

# Assign the movie titles to the index and inspect
#tfidf_df.index = movies_data['title']
#print(tfidf_df.head())

# Import cosine_similarity measure
#from sklearn.metrics.pairwise import cosine_similarity

# Create the array of cosine similarity values
#cosine_similarity_array = cosine_similarity(tfidf_df)

# Wrap the array in a pandas DataFrame
#cosine_similarity_df = pd.DataFrame(cosine_similarity_array, index=tfidf_df.index, columns=tfidf_df.index)

# Print the top 5 rows of the DataFrame
#print(cosine_similarity_df.head())
