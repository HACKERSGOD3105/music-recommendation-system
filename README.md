Emotion-Based Music Recommendation System

Overview

This project is an end-to-end AI-powered music recommendation system that suggests songs based on the user's emotional state. It combines Facial Emotion Recognition (FER) with an intelligent recommendation pipeline to deliver personalized music experiences.

The system detects user emotions using a deep learning model and maps them to relevant music using external APIs such as Last.fm and YouTube.

---

Key Features

- Real-time facial emotion detection
- Deep learning model (ResNet-34 + MTCNN) for emotion classification
- Emotion-to-music recommendation engine
- Integration with YouTube and Last.fm APIs
- Interactive frontend using Streamlit
- Fast inference and dynamic recommendations
- Secure handling of API keys using environment variables

---

System Architecture

1. Emotion Detection Pipeline

- Face detection using MTCNN
- Feature extraction using ResNet-34
- Emotion classification (Happy, Sad, Angry, Neutral, etc.)

2. Recommendation Engine

- Emotion-to-music mapping logic
- Query external APIs for song suggestions
- Ranking and filtering of relevant tracks

3. Frontend Interface

- Built using Streamlit
- Displays detected emotion
- Shows recommended songs
- Provides playback links

---

Tech Stack

Languages and Frameworks

- Python
- Streamlit

Deep Learning

- PyTorch / TensorFlow
- ResNet-34
- MTCNN

APIs

- YouTube Data API
- Last.fm API

Deployment

- Streamlit Cloud / Local deployment

---

Project Structure

emotion-music-recommender/
│
├── app.py
├── model/
├── utils/
├── recommendation/
├── requirements.txt
├── .env
└── README.md

---

Installation and Setup

1. Clone the repository

git clone https://github.com/your-username/emotion-music-recommender.git
cd emotion-music-recommender

2. Create a virtual environment

python -m venv venv
source venv/bin/activate   (Linux/Mac)
venv\Scripts\activate      (Windows)

3. Install dependencies

pip install -r requirements.txt

4. Setup environment variables

Create a ".env" file and add:

YOUTUBE_API_KEY=your_key
LASTFM_API_KEY=your_key

---

Running the Application

streamlit run app.py

---

Model Details

- Architecture: ResNet-34
- Face Detection: MTCNN
- Input: Facial image
- Output: Emotion class probabilities

Supported Emotions

- Happy
- Sad
- Angry
- Neutral
- Surprise

---

Workflow

1. Capture user image using camera input
2. Detect face using MTCNN
3. Predict emotion using the trained model
4. Map emotion to a music category
5. Fetch songs using external APIs
6. Display recommendations with playback links

---

Future Improvements

- Integration with Spotify API
- Personalized recommendations based on user history
- Emotion trend analytics dashboard
- Transformer-based emotion models
- Mobile application deployment

---

Challenges Faced

- Real-time inference optimization
- API rate limiting and filtering
- Trade-off between model accuracy and performance
- Deployment issues related to model loading

---

Key Learnings

- End-to-end machine learning system design
- Model deployment and inference optimization
- API integration in AI systems
- Building interactive ML applications

---

Author

Suriya
Aspiring Data Scientist

---

License

This project is open-source and available
