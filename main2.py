import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

# Pour montrer toutes les colonnes d'un dataframe
pd.set_option('display.max_columns', None)

# Chargement des données
data = pd.read_csv("tmdb_movies_data.csv")

# Création du dataframe
movies_df = pd.DataFrame(data)

# On enlève les colonnes que l'on ne souhaite pas exploiter
movies_df_trim = movies_df.drop(
    columns=['imdb_id', 'popularity', 'homepage', 'release_date', 'budget_adj', 'revenue_adj'])

# On crée un dataframe qui sera utilisé pour les content based recommendations
movies_data = movies_df_trim.loc[:,
              ["original_title", "cast", "director", "genres", "keywords", "overview", "vote_count", "vote_average",
               "release_year"]]

# On change le nom de la colonne du titre pour plus de clarté
movies_data.rename(columns={"original_title": "title"}, inplace=True)

# On enlève les valeurs vides
movies_data.dropna(inplace=True)

# On remplace les "|" par des virgules pour les colonnes concernées
movies_data["cast"] = movies_data["cast"].apply(lambda x: x.replace("|", ", "))
movies_data["genres"] = movies_data["genres"].apply(lambda x: x.replace("|", ", "))
movies_data["keywords"] = movies_data["keywords"].apply(lambda x: x.replace("|", ", "))
movies_data["director"] = movies_data["director"].apply(lambda x: x.replace("|", ", "))

# On extrait les valeurs uniques pour title, cast, director, genres, keywords et on les met en ordre alphabétique.
# Le résultat est un array.
unique_titles = sorted(movies_data.title.unique())
actors = movies_data['cast'].str.split(', ', expand=True).stack()
unique_actors = sorted(actors.unique())
directors = movies_data['director'].str.split(', ', expand=True).stack()
unique_directors = sorted(directors.unique())
genres = movies_data['genres'].str.split(', ', expand=True).stack()
unique_genres = sorted(genres.unique())
keywords = movies_data['keywords'].str.split(', ', expand=True).stack()
unique_keywords = sorted(keywords.unique())

# On calcule la moyenne des notes de tous les films afin de proposer les mieux notés pondérés par le nombre de votes
votes_data = movies_data.loc[:, ['vote_average', 'vote_count']]
votes_data.dropna(inplace=True)
vote_counts = votes_data['vote_count']
vote_averages = votes_data['vote_average']

#On crée un nouveau dataframe afin de recommander 30 films grâce à une analyse textuelle basée sur le descriptif du film

data_cbr=data.copy()

data_cbr['tagline']=data_cbr['tagline'].astype('str')
data_cbr['overview']=data_cbr['overview'].astype('str')

data_cbr['tagline'] = data_cbr['tagline'].fillna('')
data_cbr['description'] = data_cbr['overview'] + data_cbr['tagline']
data_cbr['description'] = data_cbr['description'].fillna('')

#On traite le dataframe pour ne garder que les termes qui nous sont utiles
#Le min_def permet d'enlever les mots qui n'apparaissent pas assez souvent, les stop_words servent à enlever les mots qui sont inutiles d'indexer ("you", "the")

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(data_cbr['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]
data_cbr = data_cbr.reset_index()
titles = data_cbr['original_title']
indices = pd.Series(data_cbr.index, index=data_cbr['original_title'])

# Streamlit
st.title('Movie recommendator :popcorn: :movie_camera:')
expander = st.expander('About')
expander.markdown("""* **Development:** This app was developed by students from the *Paris 1 Panthéon-Sorbonne* 
University MBFA program: Raquel Carvalho, Marco Chavez and Pierre-Yves Degez 
* **Data source:** [Kaggle](https://www.kaggle.com/datasets/juzershakir/tmdb-movies-dataset?resource=download) 
* **References:** [Kaggle: Content Based Recommendation by Omkaar Lavangare](https://www.kaggle.com/code/omkaarlavangare/content-based-recommendation/notebook)""")
st.header('Spending more time choosing a movie than watching it?')
st.subheader('Try our movie recommendator based on your preferences!')
st.caption('Choose as many genres as you want')

title_choice = st.multiselect("Choose your favorite movie!", unique_titles)
#actor_choice = st.multiselect("Choose your favorite actresses and actors!", unique_actors)
genre_choice = st.multiselect("Choose your favorite genres!", unique_genres)
#keywords_choice = st.multiselect("Any keywords?", unique_keywords)
number_of_films = st.slider('How many recommendations do you want?', min_value=1, max_value=10, value=5)

# Avec la ligne suivante on transforme la liste générée par le multiselect en string pour pouvoir le passer en
# argument dans la fonction build_chart
genre_choice_str = ', '.join(genre_choice)

# On filtre la data par rapport à ce qui est choisi dans les multiselect
selected_title = data_cbr[(data_cbr['titles'].isin(title_choice))]

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

get_recommendations('The Godfather').head(10)



#selected_actor = movies_data[(movies_data['cast'].isin(actor_choice))]
selected_genre = movies_data[(movies_data['genres'].isin(genre_choice))]
#selected_keywords = movies_data[(movies_data['keywords'].isin(keywords_choice))]


def build_chart(genre, percentile=0.7):
    dg = movies_data[movies_data['genres'] == genre]
    mean_votes = vote_averages.mean()
    minimum = vote_counts.quantile(percentile)
    qualified = dg[(dg['vote_count'] >= minimum) & (dg['vote_count'].notnull()) & (dg['vote_average'].notnull())][
        ['title', 'genres', 'overview', 'release_year', 'vote_count', 'vote_average']]
    qualified['weighted_avg'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + minimum) * x['vote_average']) + (
                minimum / (minimum + x['vote_count']) * mean_votes), axis=1)
    qualified = qualified.sort_values('weighted_avg', ascending=False).head(250)

    return qualified


if st.button('Go!'):
    st.subheader('Here are your results! :popcorn:')
    st.dataframe(build_chart(genre_choice_str).head(number_of_films), height = 2000)
