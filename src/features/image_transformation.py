import os
from PIL import Image
import pathlib


def reshape_image(image: Image, shape: list[int, int]):
    """Resize an image to the given shape."""
    return image.resize(shape, Image.LANCZOS)


def only_one_reshape_image(image_path: str, output_path: str, output_img_name: str, shape) -> None:
    if pathlib.Path(image_path).suffix == ".jpg":
        with open(image_path, 'r+b') as f:
            with Image.open(f) as image:
                image = reshape_image(image, shape)
                image.save(os.path.join(output_path, output_img_name), image.format)
                print("[{}/{}] Resized the images and saved as '{}'."
                      .format(1, 1, output_path))


def reshape_images(image_path: str, output_path: str, shape) -> None:
    """Reshape the images in 'image_path' and save into 'output_path'."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    images = os.listdir(image_path)
    num_im = len(images)
    for i, im in enumerate(images):
        with open(os.path.join(image_path, im), 'r+b') as f:
            with Image.open(f) as image:
                image = reshape_image(image, shape)
                image.save(os.path.join(output_path, im), image.format)
        if (i + 1) % 100 == 0:
            print("[{}/{}] Resized the images and saved into '{}'."
                  .format(i + 1, num_im, output_path))
