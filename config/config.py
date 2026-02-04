import streamlit as st

LASTFM_API_KEY = st.secrets["LASTFM_API_KEY"]
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"

YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
YOUTUBE_SEARCH_LIMIT = 10

TOP_K_SONGS = 5
LASTFM_FETCH_LIMIT = 40

