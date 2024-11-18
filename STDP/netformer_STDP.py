import torch
import torch.nn as nn

class SingleHeadAttention_cat(nn.Module):
    # the 1st var (row) is the membrane potential
    # the last column is the most recent timestep

    def __init__(self, seq_len, input_dim, emb_dim, proj_dim=None):
        super(SingleHeadAttention_cat, self).__init__()

        self.input_dim = input_dim  # n hist steps
        self.emb_dim = emb_dim
        self.seq_len = seq_len  # n var
        if proj_dim is None:
            self.proj_dim = seq_len
        else:
            self.proj_dim = proj_dim

        self.E = nn.Parameter(torch.randn(self.seq_len, self.emb_dim))
        self.query_linear = nn.Linear(in_features=self.input_dim + self.emb_dim, out_features=self.proj_dim, bias=False)
        self.key_linear = nn.Linear(in_features=self.input_dim + self.emb_dim, out_features=self.proj_dim, bias=False)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.input_dim + self.emb_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.input_dim)

    def forward(self, x):
        batch_size, _, _ = x.size()
        embedding_batch = self.E.unsqueeze(0).expand(batch_size, -1, -1)  # batch_size, seq_len, emb_dim
        x_with_embedding = torch.cat((x, embedding_batch), -1)
        # print(x_with_embedding)
        x_with_embedding = self.layer_norm1(x_with_embedding)
        scale = (self.input_dim + self.emb_dim) ** (-0.5)

        # Apply linear transformations for queries, keys, and values
        queries = self.query_linear(x_with_embedding[:, 0, :].unsqueeze(1))  # only track 1st var
        keys = self.key_linear(x_with_embedding)
        # Compute attention without softmax
        attention_weights = scale * torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, 1, sequence_length)
        # print(attention_weights.shape)

        values = x_with_embedding[:, :, :self.input_dim]
        # print(values.shape)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, values) + values[:, 0, :].unsqueeze(1)
        attended_values = self.layer_norm2(attended_values)
        # print(attended_values.shape)
        # print(attended_values[:, :, -1].shape)
        # print(attention_weights.shape)

        return attended_values[:, :, -1], attention_weights


class TransformerForecasting_cat(nn.Module):
    def __init__(self, seq_len, input_dim, emb_dim, proj_dim=None):
        super(TransformerForecasting_cat, self).__init__()
        self.attention = SingleHeadAttention_cat(seq_len, input_dim, emb_dim, proj_dim)

    def forward(self, inputs):  # inputs shape: (num_samples, sequence_length, input_dim)
        x = torch.squeeze(self.attention(inputs)[0]) # (num_samples, )
        return x


def train(dataloader, model, criterion, optimizer, max_grad_norm=None, device='cpu'):
    model.train()  # turn on train mode
    loss_all_batches = 0
    for inp_batch, target_batch in dataloader:  # inp_batch: (batch_size, seq_len, inp_dim), target_batch: (batch_size, )
        inp_batch = inp_batch.to(device)
        target_batch = target_batch.to(device)
        target_batch_pred = model(inp_batch)
        loss = criterion(target_batch_pred, target_batch)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # gradient clipping
        optimizer.step()
        loss_all_batches += loss.item()

    return loss_all_batches/len(dataloader)


def evaluate(dataloader, model, criterion, device='cpu'):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for inp_batch, target_batch in dataloader:
            inp_batch = inp_batch.to(device)
            target_batch = target_batch.to(device)
            target_batch_pred = model(inp_batch)
            total_loss += criterion(target_batch_pred, target_batch).item()

    return total_loss / len(dataloader)