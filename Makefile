.PHONY: install install-dev demo live doctor test lint

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,model]"

demo:
	python -m litearm_gpt6astra

live:
	python -m litearm_gpt6astra live

doctor:
	python -m litearm_gpt6astra doctor

test:
	pytest -q

lint:
	ruff check litearm_gpt6astra tests scripts
	ruff format --check litearm_gpt6astra tests scripts
