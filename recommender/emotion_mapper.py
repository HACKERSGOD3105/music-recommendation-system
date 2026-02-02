# emotion_mapper.py

EMOTION_TO_TAG = {
    "happy": "happy",
    "sad": "sad",
    "angry": "aggressive",
    "neutral": "chill"
}

def map_emotion_to_tag(emotion: str) -> str:
    """
    Maps detected emotion to Last.fm music tag
    """
    return EMOTION_TO_TAG.get(emotion, "chill")
