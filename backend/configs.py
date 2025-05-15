import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# To suppress hf_xet warning, set environment variable HF_HUB_DISABLE_XET=True
# or install `hf_xet` with `pip install hf_xet`

# Paths
try:
    PROJ_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    PROJ_ROOT = Path(__file__).resolve().parent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Model file paths, with environment variable overrides
RESNET_EMBEDDINGS_PATH = os.getenv(
    "RESNET_EMBEDDINGS_PATH",
    os.path.join(BASE_DIR, "models", "image_embeddings_resnet.pt")
)
RESNET_MODEL_PATH = os.getenv(
    "RESNET_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "dual_encoder_resnet.pth")
)
VIT_EMBEDDINGS_PATH = os.getenv(
    "VIT_EMBEDDINGS_PATH",
    os.path.join(BASE_DIR, "models", "image_embeddings_vit.pt")
)
VIT_MODEL_PATH = os.getenv(
    "VIT_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "dual_encoder_ViT.pth")
)

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass