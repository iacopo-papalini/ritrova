.DEFAULT_GOAL := help

.PHONY: help serve analyse analyse-photos cluster auto-assign cleanup stats css css-watch test lint migrate-paths

ENV_FILE := .env
ifeq ($(OS),Windows_NT)
ENV_FILE := .env_windows
endif

RITROVA := uv run --env-file $(ENV_FILE) ritrova
MODELS_STAMP := .models-warmed

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

serve: $(MODELS_STAMP) ## Start the web UI
	$(RITROVA) serve

# Downloads buffalo_l / yolo11m / siglip on first run so a family member's
# first manual finding doesn't stall on ~1.1 GB of weights. Delete the stamp
# to re-pull. Skipped with a warning when the `scan` extra isn't installed.
$(MODELS_STAMP):
	@$(RITROVA) warmup && touch $@ || true

analyse: ## Analyse photos and videos for people and pets
	$(RITROVA) analyse

analyse-photos: ## Analyse photos only for people and pets
	$(RITROVA) analyse --no-videos

cluster: ## Cluster all faces (humans + pets)
	$(RITROVA) cluster

auto-assign: ## Bulk-assign clusters to known persons
	$(RITROVA) auto-assign

cleanup: ## Dismiss tiny and blurry faces
	$(RITROVA) cleanup

stats: ## Show database statistics
	$(RITROVA) stats

css: ## Build Tailwind CSS (one-time)
	./tailwindcss -i src/ritrova/static/input.css -o src/ritrova/static/style.css --minify

css-watch: ## Watch and rebuild CSS on changes
	./tailwindcss -i src/ritrova/static/input.css -o src/ritrova/static/style.css --watch

test: ## Run tests
	uv run pytest tests/ -v

lint: ## Run linter + type checker
	uv run ruff check src/ tests/ && uv run mypy src/ tests/

migrate-paths: ## Rewrite DB paths to relative
	$(RITROVA) migrate-paths
