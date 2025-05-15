import os

# Base directory of the project (assumed to be Image_Search/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model file paths, with environment variable overrides
RESNET_EMBEDDINGS_PATH = os.getenv(
    "RESNET_EMBEDDINGS_PATH",
    os.path.join(BASE_DIR, "Image_Search", "models", "image_embeddings_resnet.pt")
)
RESNET_MODEL_PATH = os.getenv(
    "RESNET_MODEL_PATH",
    os.path.join(BASE_DIR, "Image_Search", "models", "dual_encoder_resnet.pth")
)
VIT_EMBEDDINGS_PATH = os.getenv(
    "VIT_EMBEDDINGS_PATH",
    os.path.join(BASE_DIR, "Image_Search", "models", "image_embeddings_vit.pt")
)
VIT_MODEL_PATH = os.getenv(
    "VIT_MODEL_PATH",
    os.path.join(BASE_DIR, "Image_Search", "models", "dual_encoder_ViT.pth")
)