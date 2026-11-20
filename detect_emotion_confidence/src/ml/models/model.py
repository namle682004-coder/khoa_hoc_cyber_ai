import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class EmotionResNet18(nn.Module):
    """
    ResNet18 based model for facial emotion recognition.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        super().__init__()

        # load backbone
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        in_features = backbone.fc.in_features

        # remove original classifier
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # initialize classifier weights
        self._init_weights()

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone (useful for fine-tuning later)"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits