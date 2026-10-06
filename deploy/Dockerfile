# Image for the Hugging Face Space (free CPU, 16 GB RAM).
# Built from the files deploy/upload_space.py uploads, not the whole repo.
FROM python:3.10-slim

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# The base image's old pip rejects some wheels on the PyTorch index
# ("inconsistent Name"), so upgrade it first.
RUN pip install --no-cache-dir --upgrade pip
# CPU-only torch: the default (CUDA) build is several GB bigger.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download bge-m3 (~2.2 GB) while building, not on the first visit.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
