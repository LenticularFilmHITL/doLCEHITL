FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y python3 python3-venv git python3-pip

# Set up Python environment
RUN python3 -m venv /opt/venv

# Activate virtual environment and upgrade pip
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /home

# Install PyTorch and other Python dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install \
    numpy \
    torch \
    matplotlib \
    argparse \
    tqdm \
    Pillow==9.5.0 \
    tensorboard \
    torchvision \
    scipy \
    segmentation-models-pytorch \
    joblib \
    scikit-image \
    albumentations \
    gradio

# Set FORCE_CUDA environment variable
ENV FORCE_CUDA="1"

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "-u", "app.py"]
