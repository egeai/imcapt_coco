import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        # self.vocab = vocab
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # caption = item.caption
        # tokens = str(caption).lower().split()
        # target = []
        # target.append(self.vocab.stoi['<start>'])
        # target.extend([self.vocab.stoi[token] for token in tokens])
        # target.append(self.vocab.stoi['<end>'])
        # target = torch.Tensor(target).long()

        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length",
                                  return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        return encoding
