import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from bs4 import BeautifulSoup
from requests import get

#st. set_page_config(layout="wide")
_ = open("loader.json")
st_lottie_html = json.load(_)

# Generate cosine similarity matrix
@st.experimental_memo
def cos_gen():
    data = pd.read_csv('Nigerian_Movies')
    data.dropna(subset='title', inplace=True)
    data.fillna('', inplace=True)
    data['show_desc_'] = data['show_desc'].str.replace('Add a plot in your language', '')
    data['stars_'] = data['stars'].str.replace(' ', '')
    data['stars_'] = data['stars_'].str.replace(',', ', ')
    data['director_'] = data['director'].str.replace(' ', '')
    data['director_'] = data['director_'].str.replace(',', ', ')
    data['text'] = data['genre']+ str(' ') + data['show_desc_']+ str(' ') + data['stars_']+ str(' ') + data['director_']
    data.reset_index(inplace=True)
    # Import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Create TfidfVectorizer object
    vectorizer = TfidfVectorizer(use_idf=True, stop_words="english")
    # Generate matrix of tf-idf vectors
    tfidf_matrix = vectorizer.fit_transform(data['text'])
    # Import cosine_similarity
    from sklearn.metrics.pairwise import linear_kernel
    # Generate cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    np.save('cosin_sim', cosine_sim)
    return cosine_sim


data = pd.read_csv('Nigerian_Movies')
data.fillna('-', inplace=True)
indices = pd.Series(data.index, index=data['title']).drop_duplicates()

# check if matrix already exists
if os.path.exists("cosine_sim.npy"):
    cosine_sim = np.load('cosine_sim.npy')
else:
    cosine_sim = cos_gen()

# list of movies
movies = data['title'].tolist()

# html html_getter
def img_getter(url:str):
    scr = 'https://illustoon.com/photo/7627.png'
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0'}
    response = get(url, headers=headers)
    tv_soup = BeautifulSoup(response.text, 'html5lib')
    for i in tv_soup.find_all('div', class_='sc-e226b0e3-4 fjlSjH'):
        image_tag = i.find('img', class_='ipc-image')

        # Check if the image tag was found and extract the source link
        if image_tag:
            image_source = image_tag['src']
            return image_source
        else: 
            pass
    return scr

# Display recommended movies
def display_movie(recommended):
    for movie in recommended:
        movie_url = data.loc[data['title']==movie, 'link'].values[0]
        st.markdown("""<p>&nbsp</p>""", unsafe_allow_html=True)
        st.markdown(display_circular_image(img_getter(movie_url)), unsafe_allow_html=True)
        st.markdown(f"""<h2 style='font-family:Courier;'><u>{movie}</u>&emsp;({int(data.loc[data['title']==movie, 'year'].values[0])})</h2>""", unsafe_allow_html=True)
        st.markdown(f"""<p style="font-size: 16px; color: red;"><b>{data.loc[data['title']==movie, 'genre'].values[0]}</b></p>""", unsafe_allow_html=True)
        st.markdown(f"""<p style="font-size: 16px;"><b>Director:</b> {data.loc[data['title']==movie, 'director'].values[0]}&emsp;&emsp;<b>Stars:</b> {data.loc[data['title']==movie, 'stars'].values[0]}</p>""", unsafe_allow_html=True)
        st.markdown(f"""<p style="font-size: 14px;">{data.loc[data['title']==movie, 'show_desc'].values[0]}</p>""", unsafe_allow_html=True)
        st.markdown(f"""[See more about this movie...]({movie_url})""", unsafe_allow_html=True)
        st.markdown('---')

def get_movie_recommendations(title, genre, cosine_sim=cosine_sim, indices=indices):
    # Get index of movie that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    recom = data['title'].iloc[movie_indices].values
    recom_genre = []
    if genre == '-':
        return recom
    else:
        for i in recom:
            if data.loc[data['title']==i, 'genre'].values[0].find(genre)==-1:
                pass
            else:
                recom_genre.append(i)
        return recom_genre


# Function to display circular images with links using HTML and CSS
def display_circular_image(image_url):
    image_style = "border: 3px solid #fff; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);"
    return f'<img src="{image_url}" alt="movie poster" style="{image_style}" width="150" height="225">'

# Main function for the Streamlit app
def main():
    st.markdown("<h1 style='font-family:Courier; text-align: center; color: red;'>🎬 Nollywood Movie Recommender 🍿</h1>", unsafe_allow_html=True)
    #st.title("🎬 Nollywood Movie Recommender 🍿")
    st.sidebar.markdown("""
    * [Dataset](https://www.kaggle.com/datasets/joshsalako/nollywood-movies-collection?trk=feed_main-feed-card_feed-article-content)
    * [Source](https://github.com/joshsalako/nollywood_recommender/)
    * [Contact me](mailto:salakojoshua1234@gmail.com)
    """)
    # Short description of the app
    st.markdown("""<p>&nbsp</p>""", unsafe_allow_html=True)
    st.markdown("""<p style="font-size: 16px;"><b>
    Welcome to the Nollywood Movie Recommender! 🚀</b></p>
    <p style="font-size: 14px;"> Select a movie you liked, set your preferences, and get personalized recommendations!
    Discover new Nollywood gems with our movie recommender! 💎
    </p>""", unsafe_allow_html=True)
    st.markdown("""<p>&nbsp</p>""", unsafe_allow_html=True)

    # Sidebar for user input
    st.markdown("""<p style="font-size: 14px;"><b>User Input: </b></p>""", unsafe_allow_html=True)

    # Movie selection
    selected_movie = st.selectbox("Select a movie you liked:", movies)

    # Genre selection
    genres = data['genre'].unique().tolist()
    genres_unique =sorted(set([words for segments in genres for words in segments.split(', ')]))
    selected_genre = st.selectbox("Select a specific genre:", genres_unique)

    # Rating selection
    #selected_rating = st.sidebar.slider("Select minimum rating:", 0.0, 5.0, 2.0, 0.1)

    # Votes selection
    #selected_votes = st.sidebar.slider("Select minimum votes:", 0, 100, 25)

    # Button to trigger movie recommendations
    if st.button("Get Recommendations"):
        # Show loading animation
        with st_lottie_spinner(st_lottie_html):
            time.sleep(2)

            # Get movie recommendations
            recommended_movies = get_movie_recommendations(selected_movie, selected_genre)

            # Show the recommendations
            st.success("Here are some movie recommendations:")
            display_movie(recommended_movies)
    st.markdown("""<p>&nbsp</p>""", unsafe_allow_html=True)

    if st.button("🎲 Get Random Movie"):
        # Show loading animation
        with st_lottie_spinner(st_lottie_html):
            time.sleep(2)
            # Get a random movie recommendation
            random_movie = np.random.choice(movies, size=1, replace=False)
            st.success(f"Random movie recommendation:")
            display_movie(random_movie)

if __name__ == "__main__":
    main()
