.PHONY: test build

build:
	docker build --no-cache --force-rm -t tnpy .

run:
	docker run --rm -i -t tnpy

test:
	python -m unittest discover -s tests -p 'test_*.py'