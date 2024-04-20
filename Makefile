greet:
	@echo "Commands:"
	@echo "  data: Fetch the datasets"
	@echo "  setup-python: Setup environment and a Jupyter kernel"
	@echo "  teardown-python: Remove the Jupyter kernel and the environment"


test:
	poetry run pytest -v


data: data/raw/lysva/field_survey.geojson
data: data/raw/trees/Birch/birch_13.las
data: data/external/newfor/Benchmark_Guidelines_NEWFOR.pdf
data/raw/lysva/field_survey.geojson:
	poetry run kaggle datasets download -p data/raw/lysva --unzip sentinel3734/tree-detection-lidar-rgb
data/raw/trees/Birch/birch_13.las:
	poetry run kaggle datasets download -p data/raw/trees --unzip sentinel3734/uav-point-clouds-of-individual-trees
data/external/newfor/Benchmark_Guidelines_NEWFOR.pdf:
	poetry run kaggle datasets download -p data/external/newfor --unzip  sentinel3734/newfor-tree-detection-benchmark


setup-python:
	poetry install --no-root
	poetry run ipython kernel install --user \
	 --name "phd" \
	 --display-name "PhD" \
	 --env PYTHONPATH $$(pwd)


teardown-python:
	jupyter kernelspec remove "phd" -y
	poetry env remove --all
