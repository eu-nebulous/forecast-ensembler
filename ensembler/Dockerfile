FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /wd

# Copy Poetry configuration files
COPY poetry.lock pyproject.toml /wd/

# Install system dependencies and Python tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        libgl1-mesa-glx \
        libx11-6 \
        curl \
    && pip install --no-cache-dir --upgrade pip --root-user-action=ignore \
    && pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only main \
    && mkdir -p /wd/models /wd/logs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and extract the morphemic-preprocessor tarball
# Create the directory and download the morphemic-preprocessor tarball
RUN mkdir -p /var/lib/morphemic && \
    curl -L -o /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0.tar.gz \
        https://gitlab.ow2.org/melodic/morphemic-preprocessor/-/archive/morphemic-rc4.0/morphemic-preprocessor-morphemic-rc4.0.tar.gz && \
    mkdir -p /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0 && \
    tar -zxf /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0.tar.gz -C /var/lib/morphemic/ && \
    cd /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0/morphemic-datasetmaker && \
    python3 setup.py install && \
    cp -R /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0/amq-message-python-library /wd/amq_message_python_library && \
    rm -rf /var/lib/morphemic

# Copy the rest of the application code
COPY app.py /wd/app.py
COPY ensembler /wd/ensembler

ENV PYTHONPATH="/wd"
# Set the default command to run the application
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# Uncomment for local testing with live reload
# CMD ["poetry", "run", "uvicorn", "app:app", "--reload"]

