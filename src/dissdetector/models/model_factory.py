import torch
import torch.nn as nn
import torchvision.models as models


MULTIMODAL_PIPELINE_MODELS = {
    "multimodal_resnet50",
    "metadata_only_mlp",
    "late_fusion_multimodal_resnet50",
}


class MultiModalResNet50(nn.Module):

    def __init__(self, num_classes: int, num_features: int = 5):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for p in backbone.parameters():
            p.requires_grad = False

        for p in backbone.layer4.parameters():
            p.requires_grad = True

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_backbone = backbone

        self.image_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, num_classes),
        )

    def forward(self, images, features):
        img_vec = self.image_backbone(images)
        img_vec = self.image_proj(img_vec)

        feat_vec = self.feature_mlp(features)

        x = torch.cat([img_vec, feat_vec], dim=1)
        return self.fusion(x)


class MetadataOnlyMLP(nn.Module):
    def __init__(self, num_classes: int, num_features: int = 5):
        super().__init__()

        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, features):
        feat_vec = self.feature_mlp(features)
        return self.classifier(feat_vec)


class LateFusionMultiModalResNet50(nn.Module):




    def __init__(self, num_classes: int, num_features: int = 5):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for p in backbone.parameters():
            p.requires_grad = False

        for p in backbone.layer4.parameters():
            p.requires_grad = True

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_backbone = backbone

        self.image_proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.image_classifier = nn.Linear(256, num_classes)

        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.feature_classifier = nn.Linear(128, num_classes)

        self.fusion_head = nn.Sequential(
            nn.Linear(num_classes * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, images, features):
        img_vec = self.image_backbone(images)
        img_vec = self.image_proj(img_vec)
        img_logits = self.image_classifier(img_vec)

        feat_vec = self.feature_mlp(features)
        feat_logits = self.feature_classifier(feat_vec)

        fused = torch.cat([img_logits, feat_logits], dim=1)
        return self.fusion_head(fused)


def is_multimodal_model(model_name: str) -> bool:

    return model_name in MULTIMODAL_PIPELINE_MODELS


def create_model(model_name: str, num_classes: int, num_features: int = 5):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "image_only_resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for p in model.parameters():
            p.requires_grad = False

        for p in model.layer4.parameters():
            p.requires_grad = True

        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "multimodal_resnet50":
        model = MultiModalResNet50(num_classes=num_classes, num_features=num_features)

    elif model_name == "metadata_only_mlp":
        model = MetadataOnlyMLP(num_classes=num_classes, num_features=num_features)

    elif model_name == "late_fusion_multimodal_resnet50":
        model = LateFusionMultiModalResNet50(num_classes=num_classes, num_features=num_features)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model
