import requests
from config.config import YOUTUBE_API_KEY, YOUTUBE_SEARCH_LIMIT

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def search_youtube_video(song: str, artist: str):
    query = f"{song} {artist} official audio"

    params = {
        "part": "snippet",
        "q": query,
        "key": YOUTUBE_API_KEY,
        "type": "video",
        "maxResults": 1
    }

    try:
        response = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])
        if not items:
            return None

        video_id = items[0]["id"]["videoId"]

        return {
            "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
            "youtube_embed": f"https://www.youtube.com/embed/{video_id}?autoplay=1"
        }

    except Exception as e:
        print(" YouTube API error:", e)
        return None
