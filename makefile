.PHONY: test build

build:
	docker build --no-cache --force-rm -t mbl .

run:
	docker run --rm -i -t -p 8501:8501 mbl

test:
	python -m unittest discover -s tests -p 'test_*.py'