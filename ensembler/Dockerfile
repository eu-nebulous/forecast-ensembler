FROM python:3.8-slim-bullseye

# Best practice: do noninteractive for apt
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /wd

# Copy Poetry files first (for caching layers)
COPY pyproject.toml poetry.lock /wd/

# Install system dependencies, update pip, install Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libgl1-mesa-glx \
    libx11-6 \
    curl \
    && pip install --no-cache-dir --upgrade pip --root-user-action=ignore \
    && pip install --no-cache-dir poetry \
    && pip install --no-cache-dir grpcio \
    && pip install --no-cache-dir python-qpid-proton \
    && poetry config virtualenvs.create false

# --- Add this step so Poetry can see your local exn/ directory ---
COPY exn /wd/exn
COPY ensembler /wd/ensembler
COPY app.py /wd/app.py

# Now install all dependencies, including the local exn path dependency
RUN poetry install --only main \
    && mkdir -p /wd/models /wd/logs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download / install your custom tarball (morphemic-preprocessor, etc.)
RUN mkdir -p /var/lib/morphemic && \
    curl -L -o /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0.tar.gz \
        https://gitlab.ow2.org/melodic/morphemic-preprocessor/-/archive/morphemic-rc4.0/morphemic-preprocessor-morphemic-rc4.0.tar.gz && \
    mkdir -p /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0 && \
    tar -zxf /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0.tar.gz -C /var/lib/morphemic/ && \
    cd /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0/morphemic-datasetmaker && \
    python3 setup.py install && \
    cp -R /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0/amq-message-python-library /wd/amq_message_python_library && \
    rm -rf /var/lib/morphemic

ENV PYTHONPATH="/wd"

CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
