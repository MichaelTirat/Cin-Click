
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""**Fonction qui renvoit le nconst en fonction du nom de la personnalité**"""

def get_nconst_from_name(name, dataframe):
    return dataframe[dataframe['primaryName'] == name]['nconst'].iloc[0]

"""**Fonction qui renvoit le nom de la personnalité en fonction du nconst**"""

# def get_name_from_nconst(nconst,dataframe):
#     return dataframe[dataframe['nconst'] == nconst]['primaryName'].iloc[0]

"""**Fonction qui renvoit les films pour laquelle la personne est connue en fonction de son nconst**

**Il faudra qu'on valide le format de sortie**
"""

def get_knownForTitles_from_nconst(nconst, dataframe):
    list_film = []
    for i in range(4):
        if type(dataframe[dataframe['nconst'] == nconst][dataframe.columns[i+2]].iloc[0]) == str :
            list_film.append(dataframe[dataframe['nconst'] == nconst][dataframe.columns[i+2]].iloc[0])
    return list_film

"""**Fonction qui renvoit les 5 meilleurs films pour lesquels la personnalité a participés**"""

def get_X_best_film_from_nconst(nconst, dataframe,n_film):
    # On classe les films par leur note pour avoir les mieux notés en premier
    dataframe.sort_values(by = 'rating', ascending = False, inplace = True)
    list_film = []
    for i in range(n_film):
        if len(dataframe[dataframe['nconst'] == nconst]) >= i+1:
            list_film.append(dataframe[dataframe['nconst'] == nconst]['tconst'].iloc[i])
    return list_film

def get_filmKnown_from_nconst_test(nconst, dataframe):
    list_film = []
    for i in range(5):
        if len(dataframe[dataframe['nconst'] == nconst]) >= i+1:
            list_film.append(dataframe[dataframe['nconst'] == nconst]['tconst'].iloc[i])
    return list_film

"""**On veut maintenant obtenir les informations du film à partir du tconst**"""

def get_info_film_from_tconst(tconst, dataframe):
    dico_film = {}
    for info in dataframe.columns:
        dico_film[info] = dataframe[dataframe['tconst'] == tconst][info].iloc[0]
    return dico_film



#    ML pairwise :

def machine_learning(dataframeKNN,  dataframePW, id_film) :
  dataframePW = dataframePW[dataframePW['tokens_title'].notna()]
  dataframePW = dataframePW[dataframePW['production_companies_name'].notna()]
  dataframePW = dataframePW.reset_index()

  list_col = ['titleType', 'Decade', 'budget', 'popularity', 'rating', 'revenue','vote_count', 'RunTime', 'Action', 'Adventure', 'Animation',
              'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy',
              'Film-Noir', 'History', 'Horror', 'Music','Musical', 'Mystery',
              'News', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War',
              'Western']

  X = dataframeKNN[list_col]
  y = dataframeKNN['tconst']

  scaler = StandardScaler().fit(X)
  X_scaled = scaler.transform(X)

  # On crée la pondération :
  weights_list = [5,1,1,1,3,1,1,1,6,6,
                  10,6,6,6,6,6,6,6,6,10,
                  6,6,6,6,6,6,6,6,6,6
                  ,6]

  # Appliquer la pondération à X_scaled
  X_scaled_weighted = X_scaled * np.array(weights_list)

  # Création du modèle et on l'entraine sur les X_scaled pondéré
  model_KNN_scaled = KNeighborsClassifier(n_neighbors=100)
  model_KNN_scaled.fit(X_scaled_weighted, y)

  # Récupération du film et on fait scale ces données
  X_data = dataframeKNN[dataframeKNN['tconst'] == id_film][list_col]
  X_data_scaled = scaler.transform(X_data)

  # On pondère X_data scaled
  X_data_scaled_weighted = X_data_scaled * np.array(weights_list)

  # On fait tourner KNN avec kneighbors
  propositionKNN_scaled = model_KNN_scaled.kneighbors(X_data_scaled_weighted)


  list_similar_KNN = []
  for film in range(1,100):
      list_similar_KNN.append(dataframeKNN.iloc[propositionKNN_scaled[1][0][film]]['tconst'])

  if id_film not in list(dataframePW['tconst'].values) :
    return list_similar_KNN[0:5]

  else :
    dataframePW = dataframePW[dataframePW['tokens_title'].notna()]
    dataframePW = dataframePW[dataframePW['production_companies_name'].notna()]
    dataframePW = dataframePW.reset_index()

        # ------------------------------------------------- Création de la combinaison
    def combined_features(row):
        return row['tokens_title']+" "+row['tokens_tagline']+" "+row['genres_x']+" "+row["production_companies_name"]+" "+str(row["Decade"])

    dataframePW['combined_features'] = dataframePW.apply(combined_features, axis = 1)

    # -------------------------------------------------- La fonction fit_transform de CountVectorizer() permet de compter le nombre de textes.
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(dataframePW['combined_features'])

    # ---------------------------------------------------  cosine_sim est un tableau numpy contenant la cosinusimilarité calculée entre deux films.
    cosine_sim = cosine_similarity(count_matrix)

    # ------------------------------------------------- Get index from titre du film

    def get_index_from(id_film2):
        return dataframePW[dataframePW['tconst'] == id_film2].index.values[0]

    movie_index = get_index_from(id_film)

    # -------------------------------------------------- Lancement de la fonction cosine_sim à partir de l'index
    similar_movies = list(enumerate(cosine_sim[movie_index]))

    # -------------------------------------------------- Sort les values du plus similaire au moins
    sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse = True)

    # Récuperer les tconsts des films à partir de l'index
    list_similar_tconst = []
    def get_tconst_from_index(index):
        return dataframePW[dataframePW.index == index]["tconst"].values[0]

    i=0
    for movies in sorted_similar_movies:
        list_similar_tconst.append(get_tconst_from_index(movies[0]))
        # print(get_tconst_from_index(movies[0]))
        i = i+1;
        if i>100:
            break

    list_similar_tconst = list_similar_tconst[1:]
    list_similar = []

    for film in list_similar_KNN :
        if film in list_similar_tconst :
            list_similar.append(film)
    return list_similar[0:5]