FROM python:3.8-slim-buster

# Install Python dependencies.
WORKDIR /wd

COPY /ensembler/poetry.lock /ensembler/pyproject.toml /wd/

RUN apt-get update && apt-get install -y build-essential g++ libgl1-mesa-glx libx11-6 \ 
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry \
    \
    && poetry install --only main && mkdir models


ADD https://gitlab.ow2.org/melodic/morphemic-preprocessor/-/archive/morphemic-rc4.0/morphemic-preprocessor-morphemic-rc4.0.tar.gz /var/lib/morphemic/

# Copy the rest of the codebase into the image
COPY ensembler/ ./

RUN cd /var/lib/morphemic/ \
    && tar -zxf morphemic-preprocessor-morphemic-rc4.0.tar.gz \
    && cd morphemic-preprocessor-morphemic-rc4.0 \
    && cd morphemic-datasetmaker && python3 setup.py install \ 
    && cd ../.. \
    && cp -R /var/lib/morphemic/morphemic-preprocessor-morphemic-rc4.0/amq-message-python-library /wd/amq_message_python_library \
    && rm -rf /var/lib/morphemic \
    && mkdir -p /wd/logs


CMD ["poetry", "run" , "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# For local tests:
# CMD ["poetry", "run" , "uvicorn", "app:app", "--reload"]


