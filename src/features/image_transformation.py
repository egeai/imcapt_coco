import os
from PIL import Image


class ImageTransformation:
    def reshape_image(self, image: Image, shape):
        """Resize an image to the given shape."""
        return image.resize(shape, Image.ANTIALIAS)

    def reshape_images(self, image_path, output_path, shape):
        """Reshape the images in 'image_path' and save into 'output_path'."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        images = os.listdir(image_path)
        num_im = len(images)
        for i, im in enumerate(images):
            with open(os.path.join(image_path, im), 'r+b') as f:
                with Image.open(f) as image:
                    image = self.reshape_image(image, shape)
                    image.save(os.path.join(output_path, im), image.format)
            if (i + 1) % 100 == 0:
                print("[{}/{}] Resized the images and saved into '{}'."
                      .format(i + 1, num_im, output_path))
