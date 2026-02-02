import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image

from models.emotion_resnet34_new import EmotionResNet34
from pipeline.emotion_to_music import emotion_to_music_pipeline

IMAGE_PATH = r"D:\annotation_processed1\sad\Sad_14.jpeg"

MODEL_PATH = r"https://huggingface.co/Suriya-31/Emotion-ResNet34/tree/main"
IMG_SIZE = 224

CLASS_NAMES = ["angry", "happy", "neutral", "sad"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


detector = MTCNN(
    image_size=IMG_SIZE,
    margin=0,
    keep_all=False,
    device=DEVICE
)

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = EmotionResNet34(
    num_classes=4,
    phase="phase2",
    verbose=False
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def detect_face(image_bgr):
   
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face = detector(rgb)

    if face is None:
        return None

    if face.dim() == 4:
        face = face[0]

    return face

def predict_emotion_from_pil(image_pil):
   

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs_tensor = F.softmax(logits, dim=1)[0]

    CLASS_NAMES = ["angry", "happy", "neutral", "sad"]

    probs = {
        CLASS_NAMES[i]: probs_tensor[i].item()
        for i in range(len(CLASS_NAMES))
    }

    
    sorted_probs = sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_emotion, top_prob = sorted_probs[0]
    second_emotion, second_prob = sorted_probs[1]

    if (top_prob - second_prob) < 0.10:
        emotion = "neutral"
        confidence = top_prob
        decision = "Mixed emotion → neutral fallback"
    else:
        emotion = top_emotion
        confidence = top_prob
        decision = "Clear top emotion"

    return emotion, confidence, probs, decision

def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f" Image not found: {IMAGE_PATH}")

    face = detect_face(image)
    if face is None:
        print(" No face detected")
        return

    face_pil = transforms.ToPILImage()(face.cpu())

    emotion, confidence, probs = predict_emotion_from_pil(face_pil)
    music_result = emotion_to_music_pipeline(emotion)

    print("\n" + "=" * 60)
    print("EMOTION PREDICTION (PRODUCTION)")
    print("=" * 60)
    print(f"Final Emotion : {emotion.upper()}")
    print(f"Confidence    : {confidence:.2%}")
    print("Decision      : Top-1 prediction")

    print("\n🎵 Recommended Songs:")
    for i, song in enumerate(music_result["songs"], 1):
        print(f"{i}. {song['song']} — {song['artist']}")
        print(f"   🔗 {song['url']}")

    print("\nAll class probabilities:")
    for name, score in probs.items():
        bar = "█" * int(score * 30)
        print(f"  {name:10s} → {score:.2%} {bar}")

    print("=" * 60)

if __name__ == "__main__":
    main()

