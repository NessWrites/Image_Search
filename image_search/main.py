import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch
import torchvision.transforms as transforms

import config
from ml_models import load_models_and_tokenizer

RESNET_EMBEDDINGS_PATH = config.RESNET_EMBEDDINGS_PATH
RESNET_MODEL_PATH = config.RESNET_MODEL_PATH
VIT_EMBEDDINGS_PATH = config.VIT_EMBEDDINGS_PATH
VIT_MODEL_PATH = config.VIT_MODEL_PATH
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_image(image):
    """Preprocess an input image for model inference.

    Args:
        image (PIL.Image.Image): The input image to preprocess.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

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
        tuple: List of (image, path, similarity) tuples and top-k similarity scores.

    Raises:
        FileNotFoundError: If model weights or embeddings file is missing.
        RuntimeError: If model weights or embeddings are incompatible.
        OSError: For I/O-related errors during file loading.
        UnidentifiedImageError: If an image file is invalid.
    """
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logging.info("Loaded fine-tuned model weights from %s", model_path)
    except (FileNotFoundError, RuntimeError, OSError) as e:
        logging.error("Failed to load model weights: %s", str(e))
        raise
    
    model.to(device)
    model.eval()

    try:
        checkpoint = torch.load(embeddings_path, map_location=device, weights_only=True)
        image_embeddings = checkpoint['embeddings'].to(device)
        image_paths = checkpoint['image_paths']
        logging.info("Loaded %d embeddings from %s", len(image_paths), embeddings_path)
    except (FileNotFoundError, RuntimeError, OSError, KeyError) as e:
        logging.error("Failed to load embeddings: %s", str(e))
        raise

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
        text_embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    
    similarities = torch.matmul(image_embeddings, text_embedding.T).squeeze()
    top_k_indices = torch.topk(similarities, k=top_k).indices

    results = []
    for idx in top_k_indices:
        image_path = image_paths[idx]
        if not os.path.exists(image_path):
            logging.warning("Image not found: %s", image_path)
            continue
        try:
            image = Image.open(image_path).convert('RGB')
            similarity = similarities[idx].item()
            results.append((image, image_path, similarity))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            logging.error("Failed to load image %s: %s", image_path, str(e))
    
    return results, similarities[top_k_indices].cpu().numpy()

def plot_metrics(resnet_metrics, vit_metrics):
    """Plot a comparison of similarity scores for ResNet and ViT models.

    Args:
        resnet_metrics (dict): Dictionary with ResNet similarity scores.
        vit_metrics (dict): Dictionary with ViT similarity scores.

    Returns:
        str: Path to the saved plot image.
    """
    ax = plt.subplots(figsize=(8, 5))[1]
    
    labels = [f'Image {i+1}' for i in range(len(resnet_metrics['similarity']))]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, resnet_metrics['similarity'], width, label='ResNet', color='skyblue')
    ax.bar(x + width/2, vit_metrics['similarity'], width, label='ViT', color='lightcoral')
    ax.set_xlabel('Top-K Images')
    ax.set_ylabel('Similarity Score')
    ax.set_title('Similarity Score Comparison: ResNet vs ViT')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('similarity_comparison.png')
    return 'similarity_comparison.png'

def main():
    """Run the Streamlit app for text-to-image search with ResNet and ViT models."""
    st.title("Fuse Machine")
    st.title("Text-to-Image Search: ResNet vs ViT Comparison")
    st.write("Enter a text query to compare search results and similarity scores from ResNet and ViT models.")

    query = st.text_input("Query", value="working on laptop in office")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

    device, tokenizer, resnet_model, vit_model = load_models_and_tokenizer()
    
    if st.button("Search"):
        if not query.strip():
            st.error("Please enter a valid query.")
            return
        
        with st.spinner("Searching..."):
            try:
                resnet_results, resnet_similarities = search_images(
                    model=resnet_model,
                    embeddings_path=RESNET_EMBEDDINGS_PATH,
                    query=query,
                    tokenizer=tokenizer,
                    device=device,
                    model_path=RESNET_MODEL_PATH,
                    top_k=top_k
                )
                
                vit_results, vit_similarities = search_images(
                    model=vit_model,
                    embeddings_path=VIT_EMBEDDINGS_PATH,
                    query=query,
                    tokenizer=tokenizer,
                    device=device,
                    model_path=VIT_MODEL_PATH,
                    top_k=top_k
                )
                
                resnet_metrics = {'similarity': resnet_similarities}
                vit_metrics = {'similarity': vit_similarities}
                
                if resnet_results or vit_results:
                    st.subheader("Search Results")
                    for i in range(max(len(resnet_results), len(vit_results))):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if i < len(resnet_results):
                                image, image_path, similarity = resnet_results[i]
                                st.image(
                                    image,
                                    caption=f"ResNet: {os.path.basename(image_path)}, Similarity: {similarity:.4f}",
                                    use_container_width=True
                                )
                            else:
                                st.write("No more results.")
                        
                        with col2:
                            if i < len(vit_results):
                                image, image_path, similarity = vit_results[i]
                                st.image(
                                    image,
                                    caption=f"ViT: {os.path.basename(image_path)}, Similarity: {similarity:.4f}",
                                    use_container_width=True
                                )
                            else:
                                st.write("No more results.")
                
                plot_path = plot_metrics(resnet_metrics, vit_metrics)
                st.subheader("Metrics Comparison")
                st.image(plot_path, caption="Comparison of Similarity Scores for ResNet and ViT", use_column_width=True)
                
                st.success(f"Found {len(resnet_results)} ResNet results and {len(vit_results)} ViT results for query: '{query}'")
            
            except (RuntimeError, ValueError, KeyError) as e:
                st.error(f"Error during search: {str(e)}")
                logging.error("Search failed: %s", str(e))

if __name__ == "__main__":
    main()