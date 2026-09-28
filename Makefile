.PHONY: install train eval inference docker clean

install:
	pip install -r requirements.txt
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

train:
	python src/train.py --config config.yaml

eval:
	python src/evaluate.py --config config.yaml

inference:
	uvicorn src.inference:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t semantic-seg-pipeline .

docker-run:
	docker run -p 8000:8000 semantic-seg-pipeline

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

lint:
	python -m flake8 src/ --max-line-length=100
