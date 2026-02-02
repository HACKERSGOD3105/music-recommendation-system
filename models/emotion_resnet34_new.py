import torch
import torch.nn as nn
from torchvision.models import resnet34


class EmotionResNet34(nn.Module):
    def __init__(
        self,
        num_classes=4,
        pretrained_path=None,
        phase="phase1",
        verbose=True
    ):
        super().__init__()

        self.phase = phase

        
        self.backbone = resnet34(weights=None)

        if pretrained_path is not None:
            state_dict = torch.load(
                pretrained_path,
                map_location="cpu"
            )

            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("fc.")
            }

            self.backbone.load_state_dict(state_dict, strict=False)

            if verbose:
                print(" Pretrained ResNet-34 weights loaded")

       
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )

        
        self._configure_trainable_layers(verbose)

    def _configure_trainable_layers(self, verbose):
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.phase == "phase1":
            # Phase-1 - FC only
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

            if verbose:
                print(" Phase-1: Training FC only")

        elif self.phase == "phase2":
            # Phase-2 - layer4 + FC
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

            for param in self.backbone.fc.parameters():
                param.requires_grad = True

            if verbose:
                print("🔓 Phase-2: Training layer4 + FC")

        else:
            raise ValueError("phase must be 'phase1' or 'phase2'")

    def forward(self, x):
        return self.backbone(x)
