import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
from PIL import Image
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader

from collections import defaultdict
from operator import itemgetter
import heapq
import os
import csv


movies = pd.read_csv("/home/gm/Documents/streamlit_project/movie.csv")
link = pd.read_csv("/home/gm/Documents/streamlit_project/link.csv")

movies['genres'] = movies['genres'].apply(lambda x : str(x).replace("|"," "))
movies['genres'] = movies['genres'].str.replace('Sci-Fi','SciFi')
movies['genres'] = movies['genres'].str.replace('Film-Noir','Noir')


# Create an object for TfidfVectorizer and this part runned on Google Colab because of Data is big and my computer couldn't finish it

#tfidf_vector = TfidfVectorizer(stop_words='english')
#tfidf_matrix = tfidf_vector.fit_transform(movies['genres'])
#sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix) 

# Sim matrix saved as a file with pickle and opened from a file.


# Function that get movie recommendations based on the cosine similarity score of movie genres
def find_poster(movie_id):
    no_poster=0
    movie_id = link['tmdbId'].where(link['movieId'] == movie_id).dropna().to_numpy(dtype ='int64')
    movie_id = movie_id[0]
    url = "https://api.themoviedb.org/3/movie/{}?api_key=c55b417414e90d9bcfa788a8d6a79d45".format(movie_id)
    try:
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
    except Exception as e:
        errnum = e.args[0]
        full_path = "/home/gm/Documents/streamlit_project/no_poster.png"
        no_poster=1
    if(no_poster != 1):
        full_path = "https://image.tmdb.org/t/p/w500" + str(poster_path)
    return full_path

# Main method that returns the recommendation
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movieId
        recommended_movie_posters.append(find_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


# Streamlit user interface design 
st.title('BAU Movie Recommender System')
header_image = Image.open('title_pic.jpg')
st.header("Content Base Recommendation")
st.image(header_image)

# Similarity matrix prepared on Google Colab and saved as a file with pickle library
similarity = pickle.load(open('/home/gm/Documents/streamlit_project/similarity.pkl','rb'))
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1,col2,col3,col4,col5 = st.columns(5)

    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])

    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
            

#------------------------Second part of the project------------------------
movies_small = pd.read_csv("/home/gm/Documents/streamlit_project/ml-latest-small/movies.csv")
ratings_dataset = pd.read_csv("/home/gm/Documents/streamlit_project/ml-latest-small/ratings.csv")
movie_list_vote = pd.read_csv("/home/gm/Documents/streamlit_project/ml-latest-small/movie_list.csv")


st.header("Collaboraive Filter with KNN")
st.text("Please give ratings for the movies from 1 to 5")
st.image(find_poster(1),width=100)
selected_movie1= st.selectbox("Toy Story (1995)",[1,2,3,4,5],key="5")
st.image(find_poster(2571),width=100)
selected_movie2 = st.selectbox('Matrix, The (1999)',options=[1,2,3,4,5],key="6")
st.image(find_poster(4369),width=100)
selected_movie3 = st.selectbox('Fast and the Furious, The (2001)',options=[1,2,3,4,5],key="7")
st.image(find_poster(1721),width=100)
selected_movie4 = st.selectbox("Titanic (1997)",options=[1,2,3,4,5],key="8")
st.image(find_poster(4185),width=100)
selected_movie5 = st.selectbox("Elvis: That's the Way It Is (1970)",options=[1,2,3,4,5],key="9")
st.image(find_poster(2028),width=100)
selected_movie6 = st.selectbox("Saving Private Ryan (1998)",options=[1,2,3,4,5],key="10")
st.image(find_poster(4896),width=100)
selected_movie7 = st.selectbox("Harry Potter and the Sorcerer's Stone (2001)",options=[1,2,3,4,5],key="11")
st.image(find_poster(53996),width=100)
selected_movie8 = st.selectbox("Transformers (2007)",options=[1,2,3,4,5],key="12")
st.image(find_poster(50872),width=100)
selected_movie9 = st.selectbox("Ratatouille (2007)",options=[1,2,3,4,5],key="13")
st.image(find_poster(69757),width=100)
selected_movie10 = st.selectbox("(500) Days of Summer (2009)",options=[1,2,3,4,5],key="14")


if st.button("Require Me Movie!"):
    ratings_dataset.at[100838,2] = selected_movie1
    ratings_dataset.at[100839,2] = selected_movie2
    ratings_dataset.at[100840,2] = selected_movie3
    ratings_dataset.at[100841,2] = selected_movie4
    ratings_dataset.at[100842,2] = selected_movie5
    ratings_dataset.at[100843,2] = selected_movie6
    ratings_dataset.at[100844,2] = selected_movie7
    ratings_dataset.at[100845,2] = selected_movie8
    ratings_dataset.at[100846,2] = selected_movie9
    ratings_dataset.at[100847,2] = selected_movie10

    
# I have done different methods but couldn't get result with the KNN methods. This trials on Google Colab. Didn't add here













