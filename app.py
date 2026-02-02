import sys
import os
import streamlit as st
from PIL import Image
import urllib.parse


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.emotion_to_music import emotion_to_music_pipeline
from inference.emotion_inference_v3 import predict_emotion_from_pil


st.set_page_config(
    page_title="Emotion-Based Music Recommender",
    layout="centered"
)


st.title("🎵 Emotion-Based Music Recommendation")
st.write("Detect your emotion and get music recommendations")


platform = st.selectbox(
    "Choose music platform:",
    ["Spotify", "YouTube"]
)


input_mode = st.radio(
    "Choose input method:",
    ("Upload Image", "Live Camera Capture")
)

image = None


if input_mode == "Upload Image":
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

elif input_mode == "Live Camera Capture":
    captured = st.camera_input("Capture image")
    if captured:
        image = Image.open(captured).convert("RGB")

def spotify_search_link(song, artist):
    query = urllib.parse.quote_plus(f"{song} {artist}")
    return f"https://open.spotify.com/search/{query}"


if image is not None:
    st.image(image, caption="Input Image", width=350)

    if st.button("🎯 Detect Emotion & Recommend Music"):
        with st.spinner("Detecting emotion and recommending music..."):

            emotion, confidence, probs, decision = predict_emotion_from_pil(image)

            music_result = emotion_to_music_pipeline(
                emotion=emotion,
                platform=platform
            )

        st.success(
            f"🎭 Detected Emotion: **{emotion.upper()}** "
            f"({confidence*100:.2f}%)"
        )
        st.info(f"Decision Logic: {decision}")

        st.subheader("📊 Emotion Probabilities")
        for emo, score in probs.items():
            st.progress(score)
            st.write(f"{emo.capitalize()}: {score*100:.2f}%")

     
        songs = music_result.get("songs", [])

        if not songs:
            st.warning("No songs found.")
        else:
          
            if platform == "YouTube" and songs[0].get("youtube_embed"):
                st.subheader("🎬 Now Playing")
                st.components.v1.iframe(
                    songs[0]["youtube_embed"],
                    height=360,
                    width=640
                )

            st.subheader("🎵 Recommended Songs")

            for i, song in enumerate(songs, 1):
                st.markdown(f"**{i}. {song['song']} — {song['artist']}**")

                if platform == "YouTube":
                    if song.get("youtube_url"):
                        st.markdown(f"[▶ Open on YouTube]({song['youtube_url']})")
                    else:
                        st.caption("⚠️ YouTube video not found")
                else:
                    spotify_url = spotify_search_link(song["song"], song["artist"])
                    st.markdown(f"[▶ Open on Spotify]({spotify_url})")

else:
    st.info("Please upload or capture an image to continue.")
