.PHONY: test build

build:
	docker build --no-cache --rm -t mbl .

run:
	docker run --rm -it -p 8501:8501 -v $(PWD)/data:/home/data mbl

exec:
	docker exec -it mbl /bin/bash

test:
	python -m unittest discover -s test -p 'test_*.py'