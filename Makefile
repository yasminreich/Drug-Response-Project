# Convenience wrappers around the Docker workflow. Everything runs inside the
# drug_response_env image -- never on the host (see CLAUDE.md).
#
# $(CURDIR) resolves the absolute repo path, so no path needs pasting by hand.

IMAGE   := drug_response_env
MOUNT   := -v "$(CURDIR)":/app -w /app
DOCKER  := docker run --rm $(MOUNT) $(IMAGE)
NB      ?= 03_baselines_comparison

.PHONY: help build test lint notebook jupyter clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build:  ## Build the Docker image
	docker build -t $(IMAGE) .

test:  ## Run the pytest suite (no 518 MB matrix needed)
	$(DOCKER) python -m pytest tests/

lint:  ## Run ruff over src/ and tests/
	$(DOCKER) ruff check .

notebook:  ## Execute a notebook in place, e.g. make notebook NB=01_initial_EDA
	$(DOCKER) jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=7200 notebooks/$(NB).ipynb

jupyter:  ## Start Jupyter on http://localhost:8888
	docker run -it --rm $(MOUNT) -p 8888:8888 $(IMAGE)

clean:  ## Remove generated artifacts (output/ is gitignored)
	rm -rf output/*.png output/*.csv output/*.npy output/*.npz output/*.json
