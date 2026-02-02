import torch
from models.emotion_resnet34_new import EmotionResNet34


NUM_CLASSES = 4
PRETRAINED_PATH = "models/best_resnet34_new_phase1_v5.pth"   # ImageNet / FER / Phase-1 path
PHASE = "phase2"   

model = EmotionResNet34(
    num_classes=NUM_CLASSES,
    pretrained_path=PRETRAINED_PATH,
    phase=PHASE
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n===== MODEL SUMMARY =====")
print("Phase:", PHASE)
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)

print("\nTrainable layers:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
