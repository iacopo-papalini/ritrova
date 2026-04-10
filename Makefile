.DEFAULT_GOAL := help

.PHONY: help serve scan scan-pets scan-videos cluster auto-assign cleanup stats css css-watch test lint migrate-paths

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

serve: ## Start the web UI
	uv run face-recog serve

scan: ## Scan photos for human faces
	uv run face-recog scan

scan-pets: ## Scan photos for dogs and cats
	uv run face-recog scan-pets

scan-videos: ## Scan videos for human faces
	uv run face-recog scan-videos

cluster: ## Cluster all faces (humans + pets)
	uv run face-recog cluster

auto-assign: ## Bulk-assign clusters to known persons
	uv run face-recog auto-assign

cleanup: ## Dismiss tiny and blurry faces
	uv run face-recog cleanup

stats: ## Show database statistics
	uv run face-recog stats

css: ## Build Tailwind CSS (one-time)
	./tailwindcss -i src/face_recog/static/input.css -o src/face_recog/static/style.css --minify

css-watch: ## Watch and rebuild CSS on changes
	./tailwindcss -i src/face_recog/static/input.css -o src/face_recog/static/style.css --watch

test: ## Run tests
	uv run pytest tests/ -v

lint: ## Run linter + type checker
	uv run ruff check src/ tests/ && uv run mypy src/ tests/

migrate-paths: ## Rewrite DB paths to relative
	uv run face-recog migrate-paths
