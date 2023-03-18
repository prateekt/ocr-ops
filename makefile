conda_dev:
	conda env remove -n ocr_ops_env
	conda env create -f conda.yaml

build:
	rm -rf dist
	rm -rf build
	rm -rf ocr_ops.egg*
	python setup.py sdist bdist_wheel

deploy:
	twine upload dist/*

test:
	nose2