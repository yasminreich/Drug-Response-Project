FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install from the FULL transitive lock, not requirements.txt. Pinning only the
# direct dependencies left pip free to resolve Jupyter's tree to releases that
# require Python >= 3.11, which broke clean builds while cached local builds
# kept passing. See the header of requirements.lock to regenerate it.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Dev tooling in a separate layer so the expensive install above stays cached.
# Without pytest/ruff in the image, the documented `make test` and `make lint`
# commands would not actually run. (requirements-dev.txt is for CI's lightweight
# test job, which never builds this image.)
RUN pip install --no-cache-dir pytest==8.3.4 ruff==0.8.6

# Register the Jupyter kernel under the environment name
RUN python -m ipykernel install --name drug_response_env --display-name "drug_response_env"

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app/notebooks"]
