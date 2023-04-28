import torch
from torch.utils.data import Dataset
from image_captioning_dataset import ImageCaptioningDataset


def collate_function(data_batch):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).
    :param data_batch: list of tuple (image, caption)
        - image: torch tensor of shape (3, 256, 256)
        - caption: torch tensor of shape (?); variable length
    :return:
        images: torch tensor of shape (batch_size, 3, 256, 256)
        targets: torch tensor of shape (batch_size, padded_length)
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length in the descending order.
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    images, captions = zip(*data_batch)

    # Merge images (3D tensors -> 4D tensors)
    # Originally, images is a list of <batch_size> number of RGB images with dimensions (3, 256, 256)
    # Turns into a single tensor of dimensions (<batch_size>, 3, 256, 256)
    images = torch.stack(tensors=images, dim=0)

    # Merge captions (1D tensors -> 2D tensors), similar to merging of images above.
    caption_lengths = [len(caption) for caption in captions]
    targets = torch.zeros(len(captions), max(caption_lengths)).long()
    for i, caption in enumerate(captions):
        end = caption_lengths[i]
        targets[i, :end] = caption[:end]
    return images, targets, caption_lengths


def get_loader(data_path, coco_json_path, vocabulary, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO Image Captioning Dataset
    coco_dataset = ImageCaptioningDataset(data_path=data_path,
                                          coco_json_path=coco_json_path,
                                          vocabulary=vocabulary,
                                          transform=transform)
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length)
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    custom_data_loader = torch.utils.data.DataLoader(dataset=coco_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     num_workers=num_workers,
                                                     collate_fn=collate_function)
    return custom_data_loader
