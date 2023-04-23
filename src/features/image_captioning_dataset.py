import os

import nltk.tokenize
import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class ImageCaptioningDataset(Dataset):
    def __init__(self, data_path, coco_json_path, vocabulary, transform: transforms = None):
        """
        Set the path for images data, captions and vocabulary wrapper.
        :param data_path: Image directory
        :param coco_json_path: coco Annotation file path
        :param vocabulary: vocabulary wrapper
        :param transform: Image transformer
        """
        self.root = data_path
        self.coco_json_data = COCO(coco_json_path)
        self.indices = list(self.coco_json_data.anns.keys())
        self.vocabulary = vocabulary
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Return one data pair (image and caption)"""
        coco_data = self.coco_json_data
        vocabulary = self.vocabulary
        annotation_id = self.indices[idx]
        caption = coco_data.anns[annotation_id]['caption']
        image_id = coco_data.anns[annotation_id]['image_id']
        image_path = coco_data.loadImgs(image_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [vocabulary('<start>')]
        caption.extend([vocabulary(token) for token in word_tokens])
        caption.append(vocabulary('<end>'))
        ground_truth = torch.Tensor(caption)
        return image, ground_truth
