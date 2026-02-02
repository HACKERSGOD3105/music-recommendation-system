import random
from recommender.lastfm_recommender import recommend_songs_from_lastfm
from recommender.youtube_recommender import search_youtube_video


def emotion_to_music_pipeline(emotion: str, platform="Spotify"):
    result = recommend_songs_from_lastfm(emotion)
    all_songs = result.get("songs", [])

    if not all_songs:
        return {"songs": []}

    random.shuffle(all_songs)
    selected = all_songs[:5]

    if platform == "YouTube":
        for song in selected:
            yt = search_youtube_video(song["song"], song["artist"])
            if yt:
                song.update(yt)

    return {"songs": selected}
