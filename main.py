import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Pour montrer toutes les colonnes d'un dataframe
pd.set_option('display.max_columns', None)

# Chargement des données
data = pd.read_csv("/Users/marcochavez/Desktop/Projet_Python/tmdb_movies_data.csv")

# Création du dataframe
movies_df = pd.DataFrame(data)

# On enlève les colonnes que l'on ne souhaite pas exploiter
movies_df_trim = movies_df.drop(
    columns=['imdb_id', 'popularity', 'homepage', 'release_date', 'budget_adj', 'revenue_adj'])

# Enlever les films en doublon
# movies_df_trim["original_title"].drop_duplicates

# On crée un dataframe qui sera utilisé pour les content based recommendations
movies_data = movies_df_trim.loc[:,
              ["original_title", "cast", "director", "genres", "keywords", "overview", "vote_count", "vote_average",
               "release_year"]]

# On change le nom de la colonne du titre pour plus de clarté
movies_data.rename(columns={"original_title": "title"}, inplace=True)

# On regarde combien il y a de valeurs nulles
# print(movies_data.isna().sum())

# On enlève les valeurs vides
movies_data.dropna(inplace=True)
# On remplace les valeurs nulles par des blanks
# movies_data.fillna(" ", inplace=True)
# print(movies_data.isna().sum())


# On remplace les "|" par des virgules pour les colonnes concernées
movies_data["cast"] = movies_data["cast"].apply(lambda x: x.replace("|", ", "))
movies_data["genres"] = movies_data["genres"].apply(lambda x: x.replace("|", ", "))
movies_data["keywords"] = movies_data["keywords"].apply(lambda x: x.replace("|", ", "))
movies_data["director"] = movies_data["director"].apply(lambda x: x.replace("|", ", "))

# On combine genres+cast+director+keywords et tagline+keywords+overview
movies_data["combined"] = movies_data["genres"] + ", " + movies_data["cast"] + ", " + movies_data["director"] + ", " + \
                          movies_data["keywords"]
# movies_data["combined2"] = movies_data["tagline"] + " " + movies_data["keywords"] + " " + movies_data["overview"]

# movies_data["title"] = movies_data["title"].str.lower()
# print(movies_data.head())
# print(movies_data.info())
# print(movies_data.shape)

# On extrait les valeurs uniques pour title, cast, director, genres, keywords et on les met en ordre alphabétique.
# Le  crésultat est un array.
unique_titles = sorted(movies_data.title.unique())
actors = movies_data['cast'].str.split(', ', expand=True).stack()
unique_actors = sorted(actors.unique())
directors = movies_data['director'].str.split(', ', expand=True).stack()
unique_directors = sorted(directors.unique())
genres = movies_data['genres'].str.split(', ', expand=True).stack()
unique_genres = sorted(genres.unique())
keywords = movies_data['keywords'].str.split(', ', expand=True).stack()
unique_keywords = sorted(keywords.unique())
# print(unique_keywords[:25])
# print(len(unique_keywords))

# Datacamp
# Instantiate the vectorizer object and transform the plot column
vectorizer = TfidfVectorizer(max_df=0.7, min_df=10)
vectorized_data = vectorizer.fit_transform(movies_data['combined'])

# Create Dataframe from TF-IDFarray
tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out())

# Assign the movie titles to the index and inspect
tfidf_df.index = movies_data['title']
# print(tfidf_df.head())

# Create the array of cosine similarity values
cosine_similarity_array = cosine_similarity(tfidf_df)

# Wrap the array in a pandas DataFrame
cosine_similarity_df = pd.DataFrame(cosine_similarity_array, index=tfidf_df.index, columns=tfidf_df.index)

# Print the top 5 rows of the DataFrame
print(cosine_similarity_df.shape)

# Proposition des meilleurs films selon des critères
# Pour pouvoir proposer des films parmi les mieux notés dans leurs catégories on va calculer les notes pondérées
# par nombre de votes pour enlever les films qui ont très peu de notes
# On crée un dataframe avec les deux colonnes qui nous intéressent et on enlève les valeurs non disponibles
# Noter bien qu'on utilise movies_df_trim et non pas movies_data
votes_data = movies_data.loc[:, ['vote_average', 'vote_count']]
votes_data.dropna(inplace=True)

vote_counts = votes_data['vote_count']
vote_averages = votes_data['vote_average']

# On calcule la moyenne des notes de tous les films
mean_votes = vote_averages.mean()

# Pour qu'un film soit inclu, il faut que le film ait plus de notes que 80% des films de la liste
# On calcule alors le minimum de notes qu'il faut qu'un film ait afin d'être pris en compte
minimum = vote_counts.quantile(0.8)

# On voit maintenant combien de films qualifient en fonction de ces critères

# movies_df_trim['year'] = pd.to_datetime(data['release_year'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# Donc ici on ne va prendre en compte que les films qui ont au minimum 207 notes et qui n'ont pas de valeur nulles
qualified = movies_data[(movies_data['vote_count'] >= minimum) & (movies_data['vote_count'].notnull()) & (
    movies_data['vote_average'].notnull())][['title', 'release_year', 'vote_count', 'vote_average', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
print(qualified.head(20))
print(qualified.shape)


# Ici on va calculer la moyenne pondérée
def moyenne_pond(x):
    c = x['vote_count']
    a = x['vote_average']
    return (c / (c + minimum) * a) + (minimum / (minimum + c) * mean_votes)


# On ajoute la colonne mp au dataframe qualified
qualified['mp'] = qualified.apply(moyenne_pond, axis=1)

# Ici on choisit d'avoir les 250 films avec les meilleurs films
qualified = qualified.sort_values('mp', ascending=False).head(250)

# Là on a les 10 films les mieux notés toutes catégories qui rentrent dans les critères que l'on a définis plus haut
print(qualified.head(10))



# Streamlit
st.title('Movie recommendator :popcorn: :movie_camera:')
expander = st.expander('About')
expander.markdown("""* **Development:** This app was developed by students from the *Paris 1 Panthéon-Sorbonne* 
University MBFA program: Raquel Carvalho, Marco Chavez and Pierre-Yves Degez 
* **Data source:** [Kaggle](https://www.kaggle.com/datasets/juzershakir/tmdb-movies-dataset?resource=download) 
* **References:** """)
st.header('Spending more time choosing a movie than watching it?')
st.subheader('Try our movie recommendator based on your preferences!')
st.caption('Choose as many as you want')

title_choice = st.multiselect("Choose your favorite movies!", unique_titles)
actor_choice = st.multiselect("Choose your favorite actresses and actors!", unique_actors)
genre_choise = st.multiselect("Choose your favorite genres!", unique_genres)
keywords_choice = st.multiselect("Any keywords?", unique_keywords)

if st.button('Go!'):
    st.write(':poop:')
