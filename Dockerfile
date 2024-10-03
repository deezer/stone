FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Install python 3.10 (Note: default in an Ubuntu 22.04 image), and other tools if needed
    python3 python3-pip python3-venv wget \
    # Install audio dependencies if needed:
    ffmpeg libsndfile-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Turn off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Configure Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install Poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==1.8.3
# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"


# Now we can install the project's Python dependencies using Poetry
WORKDIR /workspace
# COPY pyproject.toml poetry.lock ./
COPY pyproject.toml  ./
RUN poetry install --no-root

# Copy the project code and run a script
COPY ./ ./

CMD poetry run python ./__init__.py
