# Image_Search

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── app
│   ├── database.py    <- Data from third party sources.
│   ├── main.py        <- Search function
│   ├── ml_models.py   <- The final, canonical data sets for modeling.
│   └── models.py      <- The Dual Encoder Model initialization.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Image_Search and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── Image_Search   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Image_Search a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
Author: Ness Shrestha
Project Name: Image Search
Organisation: Fuse Machines
Github Link : https://github.com/NessWrites/Image_Search



"Image Search," is a text-to-image search application designed with principles inspired by the 12 Factor App methodology, ensuring scalability, maintainability, and portability. It compares the performance of ResNet and Vision Transformer (ViT) models for image retrieval based on text queries. Below is a summarized description of the project, highlighting how it aligns with 12 Factor App principles:

Functionality Overview:
**Frontend (main.py):** Built with Streamlit, it provides a user-friendly interface for entering text queries (e.g., "working on laptop in office") and selecting the number of results (top_k). It communicates with the backend via HTTP requests, displaying images from ResNet and ViT models side-by-side with similarity scores. Error handling ensures a robust user experience.
**Backend (app.py, search.py, ml_models.py, models.py):** A FastAPI service that processes queries using a dual encoder model combining vision (ResNet50 or ViT) and text (BERT) encoders. It loads precomputed image embeddings, computes similarity scores, and returns top-k images with paths and scores, enabling model comparison.
**Model Architecture (models.py):** Implements a DualEncoder class to generate normalized embeddings for images and text in a shared space, facilitating cosine similarity comparisons.
**Search Logic (search.py):** Retrieves images by encoding text queries, computing similarities with precomputed image embeddings, and returning top-k results. It includes logging for debugging and error handling.
**Configuration (configs.py):** Centralizes paths for model weights, embeddings, and data directories, using environment variables for flexibility.
**12 Factor App Principles Alignment:**
**I. Codebase:** A single codebase tracked in version control, deployed as separate frontend and backend services via Docker, ensuring consistent deployments.
**II. Dependencies:** Explicitly declared in requirements.txt for both frontend and backend, adhering to dependency isolation (e.g., streamlit, fastapi, torch).
**III. Config:** Configuration is stored in environment variables (e.g., RESNET_EMBEDDINGS_PATH, GOOGLE_CREDENTIALS_PATH) via .env files, loaded in configs.py, separating config from code for portability across environments.
**IV. Backing Services: ** The backend treats external services (e.g., Google Drive for image storage, once integrated) as attachable resources, accessible via API endpoints.
**V. Build, Release, Run:** Docker and docker-compose.yml enforce a clear separation of build (Docker images), release (tagged containers), and run (container execution) stages.
**VI. Processes:** The application runs as stateless processes in Docker containers, with no reliance on persistent local state, enabling easy scaling.
**VII. Port Binding: **Services expose ports (frontend: 8501, backend: 8000) via Docker, making them self-contained and accessible.
**VIII. Concurrency: **The FastAPI backend supports asynchronous requests, and Docker allows scaling multiple instances for concurrent workloads.
**IX. Disposability:** Containers are designed for fast startup and graceful shutdown, with logging (logging.basicConfig) for robustness.
**X. Dev/Prod Parity:** Docker ensures consistent environments across development and production, minimizing discrepancies.
**XI. Logs:** Both frontend and backend use structured logging (logging and loguru) to stream logs for monitoring and debugging.
**XII. Admin Processes:** Administrative tasks (e.g., model loading in ml_models.py) are integrated into the codebase and run within the same environment.
**Current Challenge and Planned Enhancement: **The application functions correctly but cannot display images in the Docker container due to missing local image files. You plan to integrate the Google Drive API to fetch COCO dataset images, using image paths or IDs from the search_images function (top_k_indices), aligning with the 12 Factor principle of treating backing services (Google Drive) as attachable resources.
![image](https://github.com/user-attachments/assets/4739ee52-1499-4bce-a351-f26911bf7545)
