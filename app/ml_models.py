from models import DualEncoder
import torch
from transformers import BertTokenizer


def load_models_and_tokenizer():
    """Load and initialize the tokenizer and dual encoder models for ResNet and ViT.

    Returns:
        tuple: (device, tokenizer, resnet_model, vit_model) where:
            - device (torch.device): The device to run models on (CPU or GPU).
            - tokenizer (BertTokenizer): The BERT tokenizer.
            - resnet_model (DualEncoder): The ResNet-based dual encoder model.
            - vit_model (DualEncoder): The ViT-based dual encoder model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    resnet_model = DualEncoder(vision_model='resnet')
    vit_model = DualEncoder(vision_model='vit')
    return device, tokenizer, resnet_model, vit_model