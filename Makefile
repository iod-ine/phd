greet:
	@echo "Commands:"
	@echo "  data: Fetch the datasets"
	@echo "  test: Run tests"
	@echo "  setup-python: Setup the environment and a Jupyter kernel"
	@echo "  teardown-python: Remove the Jupyter kernel and the environment"


test:
	pdm run pytest -v


data: data/raw/lysva/field_survey.geojson
data: data/raw/trees/Birch/birch_69.las
data: data/external/newfor/Benchmark_Guidelines_NEWFOR.pdf
data/raw/lysva/field_survey.geojson:
	pdm run kaggle datasets download -p data/raw/lysva --unzip sentinel3734/tree-detection-lidar-rgb
data/raw/trees/Birch/birch_69.las:
	pdm run kaggle datasets download -p data/raw/trees --unzip sentinel3734/uav-point-clouds-of-individual-trees
data/external/newfor/Benchmark_Guidelines_NEWFOR.pdf:
	pdm run kaggle datasets download -p data/external/newfor --unzip  sentinel3734/newfor-tree-detection-benchmark


setup-python:
	pdm install --no-self
	pdm run ipython kernel install --user \
	 --name "phd" \
	 --display-name "PhD" \
	 --env PYTHONPATH $$(pwd)


teardown-python:
	jupyter kernelspec remove "phd" -y
	rm -r .venv
