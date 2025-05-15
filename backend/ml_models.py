# backend/image_search/ml_models.py
import torch
from transformers import BertTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_models_and_tokenizer():
    """
    Load the ResNet and ViT dual encoder models and BERT tokenizer.

    Returns:
        tuple: (device, tokenizer, resnet_model, vit_model)
    """
    try:
        # Set device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Load BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        logging.info("Loaded BERT tokenizer")

        # Placeholder for ResNet and ViT models
        # Replace with actual model loading logic (e.g., from Hugging Face or local files)
        resnet_model = torch.nn.Module()  # Replace with actual ResNet-based dual encoder
        vit_model = torch.nn.Module()    # Replace with actual ViT-based dual encoder
        logging.info("Loaded placeholder ResNet and ViT models")

        # Move models to device
        resnet_model.to(device)
        vit_model.to(device)

        return device, tokenizer, resnet_model, vit_model

    except Exception as e:
        logging.error(f"Failed to load models or tokenizer: {str(e)}")
        raise