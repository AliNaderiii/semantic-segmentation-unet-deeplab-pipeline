.PHONY: install install-dev download train smoke-test evaluate serve benchmark test lint docker-build docker-run clean

install:
	@echo "Install a matching torch/torchvision build first; see README.md."
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

download:
	python -m src.download_data

train:
	python -m src.train --config config.yaml

smoke-test:
	python -m src.train --config config.yaml --smoke-test

evaluate:
	python -m src.evaluate --config config.yaml --checkpoint checkpoints/best.pt

serve:
	uvicorn src.inference:app --host 0.0.0.0 --port 8000

benchmark:
	python -m src.benchmark --model unet --encoder resnet18

test:
	pytest

lint:
	ruff check src tests

docker-build:
	docker build -t semantic-segmentation-pipeline .

docker-run:
	docker run --rm -p 8000:8000 -v "$(PWD)/checkpoints:/app/checkpoints:ro" semantic-segmentation-pipeline

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ reports/heldout_val_metrics.json reports/experiment_dashboard.png
