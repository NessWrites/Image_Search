import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer


class DualEncoder(nn.Module):
    """A dual encoder model for image-text matching using vision and text encoders.

    Attributes:
        temperature (float): Scaling factor for embeddings.
        vision_encoder (nn.Module): The vision encoder (ResNet or ViT).
        vision_projection (nn.Sequential): Projection head for vision features.
        text_encoder (BertModel): The BERT text encoder.
        text_projection (nn.Sequential): Projection head for text features.
    """
    def __init__(self, vision_model='resnet', embedding_dim=256, temperature=0.05):
        """Initialize the dual encoder with specified vision model and parameters.

        Args:
            vision_model (str, optional): Vision model type ('resnet' or 'vit'). Defaults to 'resnet'.
            embedding_dim (int, optional): Dimension of the output embeddings. Defaults to 256.
            temperature (float, optional): Temperature for scaling embeddings. Defaults to 0.05.

        Raises:
            ValueError: If vision_model is not 'resnet' or 'vit'.
        """
        super(DualEncoder, self).__init__()
        self.temperature = temperature
        if vision_model == 'resnet':
            self.vision_encoder = models.resnet50(pretrained=False)
            self.vision_encoder.fc = nn.Identity()
            vision_input_dim = 2048
        elif vision_model == 'vit':
            self.vision_encoder = models.vit_b_16(pretrained=False)
            self.vision_encoder.heads = nn.Identity()
            vision_input_dim = 768
        else:
            raise ValueError("vision_model must be 'resnet' or 'vit'")
        
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_input_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1)
        )
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Sequential(
            nn.Linear(768, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, images=None, input_ids=None, attention_mask=None, image_only=False):
        """Forward pass for encoding images or text.

        Args:
            images (torch.Tensor, optional): Input images for vision encoder. Defaults to None.
            input_ids (torch.Tensor, optional): Token IDs for text encoder. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask for text encoder. Defaults to None.
            image_only (bool, optional): If True, process only images. Defaults to False.

        Returns:
            torch.Tensor: Normalized embeddings for images or text.
        """
        if image_only:
            image_features = self.vision_encoder(images)
            image_embeddings = self.vision_projection(image_features)
            return nn.functional.normalize(image_embeddings, dim=-1)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        text_embeddings = self.text_projection(text_features)
        return nn.functional.normalize(text_embeddings, dim=-1)