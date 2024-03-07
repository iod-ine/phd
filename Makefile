greet:
	@echo "Commands:"
	@echo "  data: Fetch the datasets"
	@echo "  setup-python: Setup environment and a Jupyter kernel"
	@echo "  teardown-python: Remove the Jupyter kernel and the environment"


data: data/raw/lysva/field_survey.geojson
data/raw/lysva/field_survey.geojson:
	poetry run kaggle datasets download -p data/raw/lysva --unzip sentinel3734/tree-detection-lidar-rgb


setup-python:
	poetry install --no-root
	poetry run ipython kernel install --user \
	 --name "phd" \
	 --display-name "PhD" \
	 --env PYTHONPATH $$(pwd)


teardown-python:
	jupyter kernelspec remove "phd" -y
	poetry env remove --all
