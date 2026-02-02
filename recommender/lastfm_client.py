# recommender/lastfm_client.py

import requests
from config.config import LASTFM_API_KEY

LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"

def fetch_tracks_by_tag(tag: str, limit: int = 40):
    params = {
        "method": "tag.gettoptracks",
        "tag": tag,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": limit
    }

    try:
        response = requests.get(LASTFM_API_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        tracks = data["tracks"]["track"]

        songs = []
        for t in tracks:
            songs.append({
                "song": t["name"],
                "artist": t["artist"]["name"],
                "url": t["url"]
            })

        return songs

    except Exception as e:
        print(" Last.fm error:", e)
        return []
