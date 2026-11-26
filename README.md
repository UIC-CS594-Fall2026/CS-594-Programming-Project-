# Pathology RAG with PLIP + Vector DB

## Team Information

- **Aryan Dalvi**: sole contributor (all code, experiments and documentation)

## Project Description

This project answers the question, can similar past pathology images help describe a new one? We use a Hugging Face dataset for this purpose, the link of which is provided right here: [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa), this project uses [PLIP](https://huggingface.co/vinid/plip), which is a CLIP model trained on pathology data.
The project stores the embeddings in a vector database (ChromaDB). Given a new image, it retrieves the most similar known images and their Q/A text, which can then be used to describe the new image (Retrieval-Augmented Generation).

## Datasets

[PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa)

## System Architecture

_To be added._

## Implementation Details

_To be added._

## Installation and Execution

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
# Phase 2: download PathVQA and save a sample image to sample_output/
python explore_data.py

# Phase 3: generate PLIP image + text embeddings into embeddings/
python generate_embeddings.py              # first 2000 training samples
python generate_embeddings.py --limit 500  # fewer samples, faster
```

The first run downloads the dataset and the PLIP model (~600 MB).

## Evaluation

_To be added._

## AI Usage

See [AI_USAGE.md](AI_USAGE.md).

## Video Demo

_To be added._
