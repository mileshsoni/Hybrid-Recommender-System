import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd

cleaned_data_path = "data/cleaned_data.csv"

transformed_data_path = "data/transformed_data.npz"

data = pd.read_csv(cleaned_data_path)

transformed_data = load_npz(transformed_data_path)

# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Text Input
song_name = st.text_input('Enter a song name:')
st.write('You entered:', song_name)

# lowercase the input
song_name = song_name.lower()

# k recommndations
k = st.selectbox('How many recommendations do you want?', [5,10,15,20], index=1)

if st.button('Get Recommendations'):
    if (data['name'] == song_name).any():
        st.write('Recommendations for ', f"**{song_name}**")
        recommendations = content_recommendation(song_name, data, transformed_data, k)
        # display recommendations
        for ind, recommendation in recommendations.iterrows():
            song_name = recommendation['name']
            artist_name = recommendation['artist']
            
            if ind == 0:
                st.markdown("## currently playing")
                st.markdown(f"### **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            elif ind == 1:   
                st.markdown("### Next Up 🎵")
                st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            else:
                st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
    else:
        st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")
            