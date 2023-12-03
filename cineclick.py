
# IMPORTS
import streamlit as st
import time
import pandas as pd
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors
from module_function import get_nconst_from_name, get_X_best_film_from_nconst, machine_learning

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# configuration de l'app :

st.set_page_config(page_title="Cin&Click", page_icon=":movie_camera:", layout="wide")
st.markdown(f"""
                <style>
                .stApp {{background-image: url(""); 
                         background-attachment: fixed;
                         base: light;
                         background-size: cover}}
             </style>
             """, unsafe_allow_html=True)

# modeles ML et database
url= "data/full_data_base.csv"
url_famous = "data/db_famous_name.csv"
url_tnconst = "data/dataset_tconst_nconst.csv"
url_pairwise = "data/tagline_pairwise.csv"



df = pd.read_csv(url)
dfamous = pd.read_csv(url_famous)
df_tnconst = pd.read_csv(url_tnconst)
df_pair = pd.read_csv(url_pairwise)

# fonctions :

@st.cache_resource
def conversion_temps(minutes):
    heures = minutes // 60
    minutes_restantes = minutes % 60

    return heures, minutes_restantes
@st.cache_resource
def get_movie(titre_film):
    # recup tconst film à partir de la bdd:
    film_id = df[df.title == titre_film].tconst.values[0]

    return film_id


########################### recup data de l'API :
@st.cache_resource
def get_movie_info(film_id):
    response = requests.get \
        (f'https://api.themoviedb.org/3/movie/{film_id}?api_key=654f36b329773337700446be4e076cb3&language=fr-FR')

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        movie_data = response.json()
        return movie_data
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

########################### FONCTION ML pairwise :
@st.cache_resource
def propo3films(titre_film):
    # df['titleType'] = df['titleType'].factorize()[0]

    list_col = ['titleType', 'Decade', 'budget', 'popularity', 'rating', 'revenue',
                'vote_count', 'RunTime', 'Action', 'Adventure', 'Animation',
                'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy',
                'Film-Noir', 'History', 'Horror', 'Music' ,'Musical', 'Mystery',
                'News', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War',
                'Western']
    X = df[list_col]
    distanceKNN = NearestNeighbors(n_neighbors=6).fit(X)

    proposition = distanceKNN.kneighbors(df[df['title'] == titre_film][list_col])

    liste_propos = []
    for film in range(1 ,4): # On part de 1 car le premier est le film qu'on a rentré
        df_propo = df[df['tconst'] == df.iloc[proposition[1][0][film]][0]]
        for i in df_propo['tconst']:
            liste_propos.append(i)
    return liste_propos

# @st.cache_data(show_spinner="En brainstorming ...")
@st.cache_resource
def single_poster(film_id):

    response = requests.get \
        (f'https://api.themoviedb.org/3/movie/{film_id}?api_key=654f36b329773337700446be4e076cb3&language=fr-FR')
    data = response.json()
    # on récupère le lien complet grâce à l'id :
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


#
# # FRONT END
#
if __name__ == '__main__':

    with st.sidebar:
        st.image('https://filedn.eu/lHdttuiSAwVYBLWzET7NL14/Projet%20P2%20Cinema%20WCS/Streamlit/cin%26click.png', width = 250)
        st.header("En recherche d'inspiration ?")
        search_type = st.radio("", ('Par titre','Par acteur'))
        st.divider()
        colgauche, coldroite = st.columns(2)
        with colgauche:
            st.write('#')
            st.write('#')
            st.write(f"Made with :hearts: by")
        with coldroite:
            st.image("https://filedn.eu/lHdttuiSAwVYBLWzET7NL14/Projet%20P2%20Cinema%20WCS/Streamlit/BDX.png", width = 150)


    # call functions based on selectbox
    if search_type == 'Par titre':
        st.subheader("Veuillez choisir un film :movie_camera: ")
        movie_name = st.selectbox("", df.title)

        if st.button('Chercher'):
            with st.spinner('Notre équipe est en brainstorming ...'):
                time.sleep(3)
            st.markdown("Nos propositions :")

            col1, col2, col3 = st.columns(3)
            id_film = get_movie(movie_name)
            liste_propos = machine_learning(df, df_pair,id_film)

            with col1:
                try:
                    film = get_movie_info(liste_propos[0])
                    if film:
                        # Extract and print relevant information
                        title = film['title']
                        overview = film['overview']
                        release_date = film['release_date']
                        genres = [genre['name'] for genre in film['genres']]
                        lien_poster = "https://image.tmdb.org/t/p/w500/" + film['poster_path']
                        duree = film['runtime']
                        heure, minutes_restantes = conversion_temps(duree)
                        note = film['vote_average']
                        nb_votes = film['vote_count']
                        tmdb_id = film['id']

                    st.write(f"<h6 style='text-align: center; color: grey;'>{title}</h6>", unsafe_allow_html=True)
                    st.image(lien_poster, caption=f'{""}', use_column_width=False, width=300)
                    st.link_button("Fiche du film", f"https://www.themoviedb.org/movie/{tmdb_id}" , help=None, type="secondary", disabled=False, use_container_width=False)
                    st.divider()
                    st.write(f":star: {round(note,1)} /10")
                    st.write(f"Genre : {genres[0]} {',' + genres[1]}")
                    st.write(f"Année : {release_date[0:4]}")
                    st.write(f"Durée: {heure}h {minutes_restantes}mn")
                    with st.expander("Synopsis :"):
                        st.write(overview)

                except IndexError:
                    st.write(f"Les informations TMDB concernant le film '{title}' sont inexistantes")
                except Exception as e:
                    st.write(f"Une erreur s'est produite : {str(e)}")

            with col2:
                try:
                    film = get_movie_info(liste_propos[1])
                    if film:
                        # Extract and print relevant information
                        title = film['title']
                        overview = film['overview']
                        release_date = film['release_date']
                        genres = [genre['name'] for genre in film['genres']]
                        lien_poster = "https://image.tmdb.org/t/p/w500/" + film['poster_path']
                        duree = film['runtime']
                        heure, minutes_restantes = conversion_temps(duree)
                        note = film['vote_average']
                        nb_votes = film['vote_count']
                        tmdb_id = film['id']

                    st.write(f"<h6 style='text-align: center; color: grey;'>{title}</h6>", unsafe_allow_html=True)
                    st.image(lien_poster, caption=f'{""}', use_column_width=False, width=300)
                    st.link_button("Fiche du film", f"https://www.themoviedb.org/movie/{tmdb_id}", help=None,
                                   type="secondary", disabled=False, use_container_width=False)
                    st.divider()
                    st.write(f":star: {round(note, 1)} /10")
                    st.write(f"Genre : {genres[0]} {',' + genres[1]}")
                    st.write(f"Année : {release_date[0:4]}")
                    st.write(f"Durée: {heure}h {minutes_restantes}mn")
                    with st.expander("Synopsis :"):
                        st.write(overview)
                except IndexError:
                    #st.write(f"Les informations TMDB concernant le film '{title}' sont inexistantes")
                    if liste_propos[1] in list(df_pair['tconst'].values):
                        st.write(df_pair[df_pair['tconst'] == liste_propos[1]]['tagline'].iloc[0])
                    if liste_propos[1] in list(df['tconst'].values):
                        st.write(df[df['tconst'] == liste_propos[1]]['Year'].iloc[0])
                except Exception as e:
                    st.write(f"Une erreur s'est produite : {str(e)}")

            with col3:
                try:
                    film = get_movie_info(liste_propos[2])
                    if film:
                        # Extract and print relevant information
                        title = film['title']
                        overview = film['overview']
                        release_date = film['release_date']
                        genres = [genre['name'] for genre in film['genres']]
                        lien_poster = "https://image.tmdb.org/t/p/w500/" + film['poster_path']
                        heure, minutes_restantes = conversion_temps(duree)
                        note = film['vote_average']
                        nb_votes = film['vote_count']
                        tmdb_id = film['id']

                    st.write(f"<h6 style='text-align: center; color: grey;'>{title}</h6>", unsafe_allow_html=True)
                    st.image(lien_poster, caption=f'{""}', use_column_width=False, width=300)
                    st.link_button("Fiche du film", f"https://www.themoviedb.org/movie/{tmdb_id}", help=None,
                                   type="secondary", disabled=False, use_container_width=False)
                    st.divider()
                    st.write(f":star: {round(note, 1)} /10")
                    st.write(f"Genre : {genres[0]} {',' + genres[1]}")
                    st.write(f"Année : {release_date[0:4]}")
                    st.write(f"Durée: {heure}h {minutes_restantes}mn")
                    with st.expander("Synopsis :"):
                        st.write(overview)
                except IndexError:
                    st.write(f"Les informations TMDB concernant le film '{title}' sont inexistantes")
                except Exception as e:
                    st.write(f"Une erreur s'est produite : {str(e)}")


    if search_type == 'Par acteur':
        st.subheader("Veuillez choisir un nom d'Acteur :star:")
        actor_name = st.selectbox('', dfamous.primaryName)

        if st.button('Chercher'):
            st.markdown(f"Filmographie de {actor_name} :")

            col1, col2, col3 = st.columns(3)
            code_acteur = get_nconst_from_name(actor_name, dfamous)
            liste_films_acteurs = get_X_best_film_from_nconst(code_acteur, df_tnconst,3)

            with col1:
                try:
                    film = get_movie_info(liste_films_acteurs[0])
                    if film:
                        # Extract and print relevant information
                        title = film['title']
                        overview = film['overview']
                        release_date = film['release_date']
                        genres = [genre['name'] for genre in film['genres']]
                        lien_poster = "https://image.tmdb.org/t/p/w500/" + film['poster_path']
                        duree = film['runtime']
                        heure, minutes_restantes = conversion_temps(duree)
                        note = film['vote_average']
                        nb_votes = film['vote_count']
                        tmdb_id = film['id']

                    st.write(f"<h6 style='text-align: center; color: grey;'>{title}</h6>", unsafe_allow_html=True)
                    st.image(lien_poster, caption=f'{""}', use_column_width=False, width=300)
                    st.link_button("Fiche du film", f"https://www.themoviedb.org/movie/{tmdb_id}", help=None,
                                   type="secondary", disabled=False, use_container_width=False)
                    st.divider()
                    st.write(f":star: {round(note, 1)} /10")
                    st.write(f"Genre : {genres[0]} {',' + genres[1]}")
                    st.write(f"Année : {release_date[0:4]}")
                    st.write(f"Durée: {heure}h {minutes_restantes}mn")
                    with st.expander("Synopsis :"):
                        st.write(overview)
                except IndexError:
                    st.write(f"Les informations TMDB concernant le film '{title}' sont inexistantes")
                except Exception as e:
                    st.write(f"Une erreur s'est produite : {str(e)}")

            with col2:
                try:
                    film = get_movie_info(liste_films_acteurs[1])
                    if film:
                        # Extract and print relevant information
                        title = film['title']
                        overview = film['overview']
                        release_date = film['release_date']
                        genres = [genre['name'] for genre in film['genres']]
                        lien_poster = "https://image.tmdb.org/t/p/w500/" + film['poster_path']
                        duree = film['runtime']
                        heure, minutes_restantes = conversion_temps(duree)
                        note = film['vote_average']
                        nb_votes = film['vote_count']
                        tmdb_id = film['id']

                    st.write(f"<h6 style='text-align: center; color: grey;'>{title}</h6>", unsafe_allow_html=True)
                    st.image(lien_poster, caption=f'{""}', use_column_width=False, width=300)
                    st.link_button("Fiche du film", f"https://www.themoviedb.org/movie/{tmdb_id}", help=None,
                                   type="secondary", disabled=False, use_container_width=False)
                    st.divider()
                    st.write(f":star: {round(note, 1)} /10")
                    st.write(f"Genre : {genres[0]} {',' + genres[1]}")
                    st.write(f"Année : {release_date[0:4]}")
                    st.write(f"Durée: {heure}h {minutes_restantes}mn")
                    with st.expander("Synopsis :"):
                        st.write(overview)
                except IndexError:
                    st.write(f"Les informations TMDB concernant le film '{title}' sont inexistantes")
                except Exception as e:
                    st.write(f"Une erreur s'est produite : {str(e)}")


            with col3:
                try:
                    film = get_movie_info(liste_films_acteurs[2])
                    if film:
                        # Extract and print relevant information
                        title = film['title']
                        overview = film['overview']
                        release_date = film['release_date']
                        genres = [genre['name'] for genre in film['genres']]
                        lien_poster = "https://image.tmdb.org/t/p/w500/" + film['poster_path']
                        duree = film['runtime']
                        heure, minutes_restantes = conversion_temps(duree)
                        note = film['vote_average']
                        nb_votes = film['vote_count']
                        tmdb_id = film['id']

                    st.write(f"<h6 style='text-align: center; color: grey;'>{title}</h6>", unsafe_allow_html=True)
                    st.image(lien_poster, caption=f'{""}', use_column_width=False, width=300)
                    st.link_button("Fiche du film", f"https://www.themoviedb.org/movie/{tmdb_id}", help=None,
                                   type="secondary", disabled=False, use_container_width=False)
                    st.divider()
                    st.write(f":star: {round(note, 1)} /10")
                    st.write(f"Genre : {genres[0]} {',' + genres[1]}")
                    st.write(f"Année : {release_date[0:4]}")
                    st.write(f"Durée: {heure}h {minutes_restantes}mn")
                    with st.expander("Synopsis :"):
                        st.write(overview)
                except IndexError:
                    st.write(f"Les informations TMDB concernant le film '{title}' sont inexistantes")
                except Exception as e:
                    st.write(f"Une erreur s'est produite : {str(e)}")

            st.divider()

            st.markdown(f"Des films inspirés de la filmographie :")
#   Machine learning par film d'acteur

            col4, col5, col6 = st.columns(3)
            code_acteur = get_nconst_from_name(actor_name, dfamous)
            liste_films_acteurs = get_X_best_film_from_nconst(code_acteur, df_tnconst,3)
            film1 = machine_learning(df, df_pair,liste_films_acteurs[0])[0]
            film2 = machine_learning(df, df_pair,liste_films_acteurs[1])[0]
            film3 = machine_learning(df, df_pair,liste_films_acteurs[2])[0]

            with col4:
                try:
                    film = get_movie_info(film1)
                    if film:
                        # Extract and print relevant information
                        title = film['title']
                        overview = film['overview']
                        release_date = film['release_date']
                        genres = [genre['name'] for genre in film['genres']]
                        lien_poster = "https://image.tmdb.org/t/p/w500/" + film['poster_path']
                        duree = film['runtime']
                        heure, minutes_restantes = conversion_temps(duree)
                        note = film['vote_average']
                        nb_votes = film['vote_count']
                        tmdb_id = film['id']

                    st.write(f"<h6 style='text-align: center; color: grey;'>{title}</h6>", unsafe_allow_html=True)
                    st.image(lien_poster, caption=f'{""}', use_column_width=False, width=300)
                    st.link_button("Fiche du film", f"https://www.themoviedb.org/movie/{tmdb_id}", help=None,
                                   type="secondary", disabled=False, use_container_width=False)
                    st.divider()
                    st.write(f":star: {round(note, 1)} /10")
                    st.write(f"Genre : {genres[0]} {',' + genres[1]}")
                    st.write(f"Année : {release_date[0:4]}")
                    st.write(f"Durée: {heure}h {minutes_restantes}mn")
                    with st.expander("Synopsis :"):
                        st.write(overview)
                except IndexError:
                    st.write(f"Les informations TMDB concernant le film '{title}' sont inexistantes")
                except Exception as e:
                    st.write(f"Une erreur s'est produite : {str(e)}")

            with col5:
                try:
                    film = get_movie_info(film2)
                    if film:
                        # Extract and print relevant information
                        title = film['title']
                        overview = film['overview']
                        release_date = film['release_date']
                        genres = [genre['name'] for genre in film['genres']]
                        lien_poster = "https://image.tmdb.org/t/p/w500/" + film['poster_path']
                        duree = film['runtime']
                        heure, minutes_restantes = conversion_temps(duree)
                        note = film['vote_average']
                        nb_votes = film['vote_count']
                        tmdb_id = film['id']

                    st.write(f"<h6 style='text-align: center; color: grey;'>{title}</h6>", unsafe_allow_html=True)
                    st.image(lien_poster, caption=f'{""}', use_column_width=False, width=300)
                    st.link_button("Fiche du film", f"https://www.themoviedb.org/movie/{tmdb_id}", help=None,
                                   type="secondary", disabled=False, use_container_width=False)
                    st.divider()
                    st.write(f":star: {round(note, 1)} /10")
                    st.write(f"Genre : {genres[0]} {',' + genres[1]}")
                    st.write(f"Année : {release_date[0:4]}")
                    st.write(f"Durée: {heure}h {minutes_restantes}mn")
                    with st.expander("Synopsis :"):
                        st.write(overview)
                except IndexError:
                    st.write(f"Les informations TMDB concernant le film '{title}' sont inexistantes")
                except Exception as e:
                    st.write(f"Une erreur s'est produite : {str(e)}")


            with col6:
                try:
                    film = get_movie_info(film3)
                    if film:
                        # Extract and print relevant information
                        title = film['title']
                        overview = film['overview']
                        release_date = film['release_date']
                        genres = [genre['name'] for genre in film['genres']]
                        lien_poster = "https://image.tmdb.org/t/p/w500/" + film['poster_path']
                        duree = film['runtime']
                        heure, minutes_restantes = conversion_temps(duree)
                        note = film['vote_average']
                        nb_votes = film['vote_count']
                        tmdb_id = film['id']

                    st.write(f"<h6 style='text-align: center; color: grey;'>{title}</h6>", unsafe_allow_html=True)
                    st.image(lien_poster, caption=f'{""}', use_column_width=False, width=300)
                    st.link_button("Fiche du film", f"https://www.themoviedb.org/movie/{tmdb_id}", help=None,
                                   type="secondary", disabled=False, use_container_width=False)
                    st.divider()
                    st.write(f":star: {round(note, 1)} /10")
                    st.write(f"Genre : {genres[0]} {',' + genres[1]}")
                    st.write(f"Année : {release_date[0:4]}")
                    st.write(f"Durée: {heure}h {minutes_restantes}mn")
                    with st.expander("Synopsis :"):
                        st.write(overview)
                except IndexError:
                    st.write(f"Les informations TMDB concernant le film '{title}' sont inexistantes")
                except Exception as e:
                    st.write(f"Une erreur s'est produite : {str(e)}")
