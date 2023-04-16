#!/usr/bin/env python3

import os
import pickle
import time

import hydra
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import logging

# from models.train_model import Train
from src.data.make_dataset import data_ingest_flow
from src.features.build_vocabulary import Vocabulary
from src.features.image_transformation import reshape_images

# from src.models.valid_model import Validation

# from fastai.vision.all import *
# from fastai.text.all import *
# from fastai.collab import *
# from fastai.tabular.all import *


logging.set_verbosity_error()


# def fastai_try(path: str):
#    dls = ImageDataLoaders.from_folder(path='../data/processed/train/images/', verbose=True)
#    print(dls)


def make_dir(path_of_folder: str) -> bool:
    is_exist = os.path.exists(path_of_folder)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(path_of_folder)
        return True


def plot_history(history):
    train_history = history['train']
    dev_history = history['dev']
    plt.plot(list(range(1, len(train_history) + 1)), train_history, label="train loss")
    plt.plot(list(range(1, len(dev_history) + 1)), dev_history, label="dev loss")
    plt.xticks(list(range(1, len(train_history) + 1)))
    plt.xlabel("epoch")
    plt.legend()


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def training(model, train_dataloader, val_dataloader, device, cfg: DictConfig):
    model.to(device)
    # set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                        patience=cfg.params.patience,
                                                        factor=cfg.params.factor)
    best_val_score = float('inf')
    train_history = []
    val_history = []
    number_of_bad_epochs = 0

    print("Learning phase")
    print("Used device: ", device)
    print("------------------------------")

    for epoch in range(1, cfg.params.max_epochs + 1):
        print("Epoch {:03d}/{:03d}".format(epoch, cfg.params.max_epochs))

        if number_of_bad_epochs >= cfg.params.max_bad_epochs:
            print(
                "Epoch {:03d}/{:03d}: exiting training after too many bad epochs.".format(epoch, cfg.params.max_epochs))
            torch.save(model.state_dict(), "final.pt")
            break
        else:
            epoch_start_time = time.time()

            train = Train(model=model, train_loader=train_dataloader,
                          optimizer=optimizer,
                          epoch=epoch,
                          max_epochs=cfg.params.max_epochs,
                          device=device)

            epoch_train_loss = train.run()
            validate = Validation(model=model, val_dataloader=val_dataloader,
                                  epoch=epoch,
                                  max_epochs=cfg.params.max_epochs)
            epoch_val_score = validate.run()

            duration = time.time() - epoch_start_time

            lr_scheduler.step(epoch_val_score)
            train_history.append(epoch_train_loss)

            val_history.append(epoch_val_score)

            if epoch_val_score < best_val_score:
                number_of_bad_epochs = 0
                best_val_score = epoch_val_score
                torch.save(model.state_dict(), "best.pt")
                print("Finished epoch {:03d}/{:03d} - Train loss: {:.7f} - Valid loss: {:.7f} - SAVED (NEW) BEST "
                      "MODEL. Duration: {:.3f} s".format(epoch,
                                                         cfg.params.max_epochs,
                                                         epoch_train_loss,
                                                         epoch_val_score,
                                                         duration))
            else:
                number_of_bad_epochs += 1
                print("Finished epoch {:03d}/{:03d} - Train loss: {:.7f} - Valid loss: {:.7f} - NUMBER OF BAD "
                      "EPOCH.S: {}. Duration: {:.3f} s".format(epoch,
                                                               cfg.params.max_epochs,
                                                               epoch_train_loss,
                                                               epoch_val_score,
                                                               number_of_bad_epochs,
                                                               duration))

    history = {'train': train_history, 'dev': val_history}
    return history


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Build vocabulary
    # vocabs = Vocabulary([], cfg).start_flow()

    # set device to cuda
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # read and prepare dataset structure
    data_ingest_flow(cfg)

    # build vocabulary
    vocab = Vocabulary(json_path=cfg.paths.raw.unzipped_annotations_subfolder + '/annotations/captions_train2014.json',
                       conf=cfg)
    vocab = vocab.build_vocab_flow()

    # with open(cfg.paths.raw.vocab_path, 'wb') as f:
    #    pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))

    image_path = cfg.paths.train.images_raw_train_train_data
    output_path = cfg.paths.train.processed_images_data
    image_shape = [256, 256]
    reshape_images(image_path, output_path, image_shape)

    # print("Saved the vocabulary wrapper to '{}'".format(cfg.paths.raw.vocab_path))

    """
    # initialize, load model and processor from pre-trained
    processor = AutoProcessor.from_pretrained(cfg.params.microsoft_pretrained)
    model = AutoModelForCausalLM.from_pretrained(cfg.params.microsoft_pretrained).to("cuda")
    # load train, valid and test datasets
    root = "../data/processed/"
    train_dataset = load_dataset("imagefolder", data_dir=root, split="train")
    val_dataset = load_dataset("imagefolder", data_dir=root, split="validation")
    test_dataset = load_dataset("imagefolder", data_dir=root, split="test")
    # creating train and validate Dataset and DataLoader
    train_ds = ImageCaptioningDataset(dataset=train_dataset, processor=processor)
    val_ds = ImageCaptioningDataset(dataset=val_dataset, processor=processor)
    train_dataloader = torch.utils.data.DataLoader(train_ds, collate_fn=collate_fn, batch_size=cfg.params.batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_ds, collate_fn=collate_fn, batch_size=cfg.params.batch_size)

    history = training(model, train_dataloader, val_dataloader, device=device, cfg=cfg)
    # plot the history
    plot_history(history)
    """
    # fastai_try(path='../data/processed/train/')


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[1]
    # mkdir data folder
    make_dir("data")
    # mkdir processed sub folder
    make_dir("data/processed")
    # mkdir raw sub folder
    make_dir("data/raw")
    # call the main function
    main()
