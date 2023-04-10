from tqdm import tqdm


class Train:

    def __init__(self, model, train_loader, optimizer, epoch, max_epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.device = device

    @staticmethod
    def learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def run(self) -> float:
        self.model.train()
        number_of_batches = len(self.train_loader)
        tqdm_obj = tqdm(self.train_loader, total=number_of_batches)
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm_obj):
            outputs = self.model(
                input_ids=batch['input_ids'].squeeze().to(self.device),
                attention_mask=batch['attention_mask'].squeeze().to(self.device),
                pixel_values=batch['pixel_values'].squeeze().to(self.device),
                return_loss=True)
            loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
            loss.backward()
            self.optimizer.step()
            tqdm_obj.set_postfix(
                batch="{}/{}".format(i + 1, number_of_batches),
                train_loss=loss.item(),
                lr=self.learning_rate(self.optimizer)
            )
        epoch_loss = epoch_loss / number_of_batches
        return epoch_loss
