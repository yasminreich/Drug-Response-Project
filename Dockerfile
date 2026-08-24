FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dev tooling in a separate layer so the expensive torch install above stays
# cached. Without pytest/ruff in the image, the documented `make test` and
# `make lint` commands would not actually run.
COPY requirements-dev.txt .
RUN pip install --no-cache-dir pytest==8.3.4 ruff==0.8.6

# Register the Jupyter kernel under the environment name
RUN python -m ipykernel install --name drug_response_env --display-name "drug_response_env"

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app/notebooks"]
