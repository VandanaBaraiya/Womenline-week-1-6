.PHONY: install run test fmt docker-build docker-run

install:
	pip install -r requirements.txt

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

docker-build:
	docker build -t summarization-engine:latest .

docker-run:
	docker run -p 8000:8000 summarization-engine:latest
