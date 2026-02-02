# recommender/lastfm_recommender.py

import random
from recommender.emotion_mapper import map_emotion_to_tag
from recommender.lastfm_client import fetch_tracks_by_tag
from config.config import TOP_K_SONGS, LASTFM_FETCH_LIMIT


def recommend_songs_from_lastfm(emotion: str):
    """
    Main recommender entry point (Last.fm)
    - Fetches many songs
    - Randomly selects TOP_K_SONGS
    """

    tag = map_emotion_to_tag(emotion)

    all_songs = fetch_tracks_by_tag(
        tag=tag,
        limit=LASTFM_FETCH_LIMIT
    )

    if not all_songs:
        return {
            "emotion": emotion,
            "tag": tag,
            "songs": []
        }

    # 🔥 RANDOM SELECTION (FIX)
    selected_songs = random.sample(
        all_songs,
        min(TOP_K_SONGS, len(all_songs))
    )

    return {
        "emotion": emotion,
        "tag": tag,
        "songs": selected_songs
    }
