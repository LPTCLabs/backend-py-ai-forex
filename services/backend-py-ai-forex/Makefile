install:
	poetry run pip install --upgrade pip
	poetry install
update:
	poetry run pip install --upgrade pip
	poetry update
build:
	rm -rf package
	poetry build
	poetry run pip install --upgrade pip
	poetry run pip install --force-reinstall -t package dist/*.whl
test:
	poetry run pytest --cov=src tests/ --cov-config=.coveragerc --cov-report xml:tests/coverage_report.xml --junitxml=tests/tests_report.xml -o junit_family=xunit2
build-docker:
	docker build -t backend-py-ai-nowcasting-forex . -f Dockerfile
qa:
	poetry run isort src tests
	poetry run black src tests
	poetry run mypy src tests
	poetry run flake8
