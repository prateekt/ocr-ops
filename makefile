conda_dev:
	conda env remove -n ocr_ops_env
	conda env create -f conda.yaml

build:
	rm -rf dist
	hatch build

publish:
	hatch publish

test:
	nose2

clean:
	rm -rf dist
	rm -rf .pytest_cache