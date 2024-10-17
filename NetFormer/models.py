import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from NetFormer.layers import Attention



####################################################################################################
# NetFormer model for connectivity-constrained simulation data
####################################################################################################

class Base_sim(pl.LightningModule):
    def __init__(self,) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=6,
                ),
                "monitor": "VAL_sum_loss",
            }
        elif self.hparams.scheduler == "cycle":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.hparams.learning_rate / 2,
                    max_lr=self.hparams.learning_rate * 2,
                    cycle_momentum=False,
                ),
                "interval": "step",
            }
        else:
            print("No scheduler is used")
            return [optimizer]

        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, neuron_level_attention = self(x)
        
        pred = y_hat
        target = y

        loss = F.mse_loss(pred, target, reduction="mean")
        self.log("TRAIN_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, neuron_level_attention = self(x)
        
        pred = y_hat
        target = y

        loss = F.mse_loss(pred, target, reduction="mean")
        self.log("VAL_loss", loss)

        result = torch.stack([pred.cpu().detach(), target.cpu().detach()], dim=1)
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat, neuron_level_attention = self(x)

        return y_hat, y, neuron_level_attention
    

class NetFormer_sim(Base_sim):
    def __init__(
        self,
        model_random_seed=42,
        neuron_num=200,
        num_cell_types=4,
        window_size=200,
        learning_rate=1e-4,
        scheduler="plateau",
        predict_window_size=1,
        attention_activation = "none", # "softmax" or "sigmoid" or "tanh", "none"
        weight_decay = 0,
        out_layer=False,
        dim_E=100,
    ):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(model_random_seed)

        self.predict_window_size = predict_window_size
        self.neuron_num = neuron_num

        dim_X = window_size - predict_window_size

        # Attention

        self.embedding_table = nn.Embedding(
            num_embeddings=neuron_num, embedding_dim=dim_E
        )

        self.layer_norm = nn.LayerNorm(dim_X+dim_E)

        self.attentionlayer = Attention(
            dim_X=dim_X,
            dim_E=dim_E,
            activation=attention_activation,
        )

        self.layer_norm2 = nn.LayerNorm(dim_X)

        self.out_layer = out_layer
        if out_layer == True:
            self.out = nn.Linear(dim_X, predict_window_size, bias=False)

    def forward(self, x):

        idx = torch.arange(x.shape[1]).to(x.device)
        e = self.embedding_table(idx)
        e = e.repeat(x.shape[0],1,1)  # (m, e) to (b, m, e)

        x_e = self.layer_norm(torch.cat((x, e), dim=-1))
        # Split x and e
        x = x_e[:, :, :x.shape[-1]]
        e = x_e[:, :, x.shape[-1]:]

        x, attn = self.attentionlayer(x, e)
        x = self.layer_norm2(x)

        if self.out_layer:
            return self.out(x), attn
        else:
            return x[:, :, -1*self.predict_window_size:], attn