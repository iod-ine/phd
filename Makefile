greet:
	@echo "Commands:"
	@echo "  data: Fetch the datasets"


data: data/raw/lysva/field_survey.geojson
data/raw/lysva/field_survey.geojson:
	poetry run kaggle datasets download -p data/raw/lysva --unzip sentinel3734/tree-detection-lidar-rgb
