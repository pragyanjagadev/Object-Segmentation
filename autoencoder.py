import pytorch_lightning as pl
import torch.nn as nn


class AutoEncoder(pl.LightningModule):

    def __init__(self, input_shape, representation_size):
        super().__init__()

        self.save_hyperparameters()  # Saves the hyperparams -- input_shape, representation_size

        self.input_shape = input_shape
        self.representation_size = representation_size

        # Calculate the flattened size
        flattened_size = 1
        for x in self.input_shape:
            flattened_size *= x

        self.flattened_size = flattened_size

        # Initialise the Dense Layers
        self.input_to_representation = nn.Linear(self.flattened_size, self.representation_size)
        self.representation_to_output = nn.Linear(self.representation_size, self.flattened_size)

    def forward(self, image_batch):
        ## ENCODING
        # image_batch: [batch_size, ...] -- Other dimensions are the input_shape
        flattened = image_batch.view(-1, self.flattened_size)
        # flattened: [batch_size, flattened_size]
        representation = F.relu(self.input_to_representation(flattened))
        # representation: [batch_size, representation_size]

        ## DECODING
        flat_reconstructed = F.relu(self.representation_to_output(representation))
        # flat_reconstructed: [batch_size, flattened_size]
        reconstructed = flat_reconstructed.view(-1, *self.input_shape)
        # reconstructed is same shape as image_batch

        return reconstructed

    def training_step(self, batch, batch_idx):
        batch_images = batch[0]
        # Get the reconstructed images
        reconstructed_images = self.forward(batch_images)
        # Calculate loss
        batch_loss = F.mse_loss(reconstructed_images, batch_images)

        # store the result
        result = pl.TrainResult(minimize=batch_loss)
        result.batch_loss = batch_loss
        result.log('train_loss', batch_loss, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        batch_images = batch[0]
        # Get the reconstructed images
        reconstructed_images = self.forward(batch_images)
        # Calculate loss
        batch_loss = F.mse_loss(reconstructed_images, batch_images)

        # store the result
        result = pl.EvalResult(checkpoint_on=batch_loss)
        result.batch_loss = batch_loss

        return result

    def test_step(self, batch, batch_idx):
        batch_images = batch[0]
        # Get the reconstructed images
        reconstructed_images = self.forward(batch_images)
        # Calculate loss
        batch_loss = F.mse_loss(reconstructed_images, batch_images)

        # store the result
        result = pl.EvalResult(checkpoint_on=batch_loss)
        result.batch_loss = batch_loss

        return result

    def validation_end(self, outputs):
        # Take mean of all batch losses
        avg_loss = outputs.batch_loss.mean()
        result = pl.EvalResult(checkpoint_on=avg_loss)
        result.log('val_loss', avg_loss, prog_bar=True)
        return result

    def test_epoch_end(self, outputs):
        # Take mean of all batch losses
        avg_loss = outputs.batch_loss.mean()
        result = pl.EvalResult()
        result.log('test_loss', avg_loss, prog_bar=True)
        return result

    def configure_optimizers(self):
        return optim.Adam(self.parameters())
