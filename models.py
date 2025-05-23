import torch
import torch.nn as nn
import timm
from transformers import ViTForImageClassification, ViTConfig
import torchvision.models as models

class VanillaViT(nn.Module):
    """Simple Vision Transformer from scratch"""
    def __init__(self, num_classes=2, img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        
    def forward(self, x):
        return self.vit(x)

class PretrainedViT(nn.Module):
    """Pre-trained Vision Transformer"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.vit(x)

class EnsembleViT(nn.Module):
    """Ensemble model: VGG16 + InceptionV3 + DenseNet201 -> ViT"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Feature extractors (remove final classifier layers)
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Identity()
        
        self.inception = models.inception_v3(pretrained=True)
        self.inception.fc = nn.Identity()
        
        self.densenet = models.densenet201(pretrained=True)
        self.densenet.classifier = nn.Identity()
        
        # Feature dimensions
        vgg_features = 25088
        inception_features = 2048
        densenet_features = 1920
        
        total_features = vgg_features + inception_features + densenet_features
        
        # Vision Transformer for final classification
        self.feature_projection = nn.Linear(total_features, 768)
        self.vit_classifier = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True),
            num_layers=6
        )
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # Extract features from each backbone
        vgg_feat = self.vgg16(x).flatten(1)
        
        # Handle inception auxiliary output
        if self.training:
            inception_feat, _ = self.inception(x)
        else:
            inception_feat = self.inception(x)
        inception_feat = inception_feat.flatten(1)
        
        densenet_feat = self.densenet(x).flatten(1)
        
        # Concatenate features
        combined_features = torch.cat([vgg_feat, inception_feat, densenet_feat], dim=1)
        
        # Project to transformer dimension
        projected_features = self.feature_projection(combined_features)
        projected_features = projected_features.unsqueeze(1)  # Add sequence dimension
        
        # Apply transformer
        transformer_output = self.vit_classifier(projected_features)
        pooled_output = transformer_output.mean(dim=1)  # Global average pooling
        
        # Final classification
        output = self.classifier(pooled_output)
        return output

def get_model(model_type, num_classes=2):
    """Factory function to get models"""
    if model_type == 'vanilla_vit':
        return VanillaViT(num_classes=num_classes)
    elif model_type == 'pretrained_vit':
        return PretrainedViT(num_classes=num_classes)
    elif model_type == 'ensemble_vit':
        return EnsembleViT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")