from tqdm import tqdm


class Validation:

    def __init__(self, model, val_dataloader, epoch, max_epochs):
        self.model = model
        self.val_dataloader = val_dataloader
        self.epoch = epoch
        self.max_epochs = max_epochs

    def run(self) -> float:
        self.model.eval()
        number_of_batches = len(self.val_dataloader)
        tqdm_obj = tqdm(self.val_dataloader, total=number_of_batches)
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm_obj):
            outputs = self.model(
                input_ids=batch['input_ids'].squeeze(),
                attention_mask=batch['attention_mask'].squeeze(),
                pixel_values=batch['pixel_values'].squeeze(),
                return_loss=True)
            loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
            epoch_loss += loss.item()
            tqdm_obj.set_postfix(
                batch="{}/{}".format(i+1, number_of_batches),
                dev_loss=loss.item()
            )
        epoch_loss = epoch_loss / number_of_batches
        return epoch_loss
