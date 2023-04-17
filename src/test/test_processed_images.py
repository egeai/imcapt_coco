import os
import unittest
from pathlib import Path
import pytest
from PIL import Image
from src.features.image_transformation import only_one_reshape_image

PROJ_DIR = Path(__file__).parent.parent.parent

input_image_path = str(PROJ_DIR) + "/data/raw/all_data/train/train2014/COCO_train2014_000000000009.jpg"
output_image_path = str(PROJ_DIR) + "/data/processed/"
output_img_name = "COCO_train2014_000000000009.jpg"


def process_sample_image():
    print("process_sample_image")
    # Process sample image. That will be in data/processed/train folder
    if not os.path.exists(os.path.join(output_image_path, output_img_name)):
        only_one_reshape_image(image_path=input_image_path,
                               output_path=output_image_path,
                               output_img_name="COCO_train2014_000000000009.jpg",
                               shape=[256, 256])
    return Image.open(os.path.join(output_image_path, output_img_name))


class TestProcessedImages(unittest.TestCase):

    @pytest.mark.run(order=1)
    def test_processed_image_height(self):
        print("test_processed_image_height")
        img: Image = process_sample_image()
        w, h = img.size
        self.assertEqual(256, h)
        img.close()
        # remove test image
        os.remove(os.path.join(output_image_path, output_img_name))

    @pytest.mark.run(order=2)
    def test_processed_image_width(self):
        print("test_processed_image_width")
        img: Image = process_sample_image()
        w, h = img.size
        self.assertEqual(256, w)
        img.close()

