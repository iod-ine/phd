from src.models.pointnet.semantic_segmentation import PointNet2Segmentor

# TODO: Add tests for parameter changing after step, loss decreasing after step.

def test_pointnet_segmento_initializes():
    PointNet2Segmentor(n_features=4)
