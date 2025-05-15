import logging
import os
import torch
from PIL import Image, UnidentifiedImageError
import configs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_images(model, embeddings_path, query, tokenizer, device, model_path, top_k=3):
    """Search for images based on a text query using a dual encoder model.

    Args:
        model (nn.Module): The dual encoder model for image-text matching.
        embeddings_path (str): Path to the precomputed image embeddings file.
        query (str): The text query for searching images.
        tokenizer (transformers.BertTokenizer): Tokenizer for text processing.
        device (torch.device): Device to run the model on (CPU or GPU).
        model_path (str): Path to the model's pretrained weights.
        top_k (int, optional): Number of top results to return. Defaults to 3.

    Returns:
        tuple: List of (image, path, similarity) tuples and top-k similarity scores, or (None, None) if files are missing.
    """
    if not os.path.exists(model_path):
        logging.warning("Model file not found: %s", model_path)
        return None, None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logging.info("Loaded fine-tuned model weights from %s", model_path)
    except (FileNotFoundError, RuntimeError, OSError) as e:
        logging.warning("Failed to load model weights from %s: %s", model_path, str(e))
        return None, None

    model.to(device)
    model.eval()

    if not os.path.exists(embeddings_path):
        logging.warning("Embeddings file not found: %s", embeddings_path)
        return None, None

    try:
        checkpoint = torch.load(embeddings_path, map_location=device, weights_only=True)
        image_embeddings = checkpoint['embeddings'].to(device)
        image_paths = checkpoint.get('image_paths', [])
        logging.info("Loaded %d embeddings from %s", len(image_paths), embeddings_path)
        if not image_paths:
            logging.warning("No image paths found in embeddings file: %s", embeddings_path)
    except (FileNotFoundError, RuntimeError, OSError, KeyError) as e:
        logging.warning("Failed to load embeddings from %s: %s", embeddings_path, str(e))
        return None, None

    tokens = tokenizer(
        query,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        logging.info("Computing text embedding for query: %s", query)
        text_embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        logging.info("Text embedding shape: %s", text_embedding.shape)

    similarities = torch.matmul(image_embeddings, text_embedding.T).squeeze()
    top_k_indices = torch.topk(similarities, k=min(top_k, len(image_paths))).indices

    results = []
    if not image_paths:
        logging.warning("No images available to process for query: %s", query)
        return results, torch.tensor([]).cpu().numpy()

    for idx in top_k_indices:
        image_path = image_paths[idx]
        if not os.path.exists(image_path):
            logging.warning("Image not found: %s", image_path)
            continue
        try:
            image = Image.open(image_path).convert('RGB')
            similarity = similarities[idx].item()
            results.append((image, image_path, similarity))
            logging.info("Loaded image: %s with similarity: %.4f", image_path, similarity)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            logging.warning("Failed to load image %s: %s", image_path, str(e))
            continue

    if not results:
        logging.info("No valid images found for query: %s", query)

    return results, similarities[top_k_indices].cpu().numpy()