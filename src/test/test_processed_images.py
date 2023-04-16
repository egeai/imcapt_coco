import unittest

from PIL import Image
from hydra import compose, initialize, initialize_config_module


class TestProcessedImages(unittest.TestCase):

    def test_processed_image_width(self):
        with initialize(version_base="1.3.2", config_path="../conf"):
            cfg = compose(config_name="config")  # overrides=["app.user=test_user"]

            im = Image.open(cfg.tests.paths.processed_image)
            w, h = im.size
            self.assertEqual(256, w)

    def test_processed_image_height(self):
        with initialize(version_base="1.3.2", config_path="../conf"):
            cfg = compose(config_name="config")  # overrides=["app.user=test_user"]

            im = Image.open(cfg.tests.paths.processed_image)
            w, h = im.size
            self.assertEqual(256, h)
