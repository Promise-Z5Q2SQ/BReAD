import torch
from torch import nn
from .config import *
import torch.nn.functional as F


class eeg_encoder_mlp(nn.Module):
    """
    MLP for frequency domain
    """
    def __init__(self, config: Config_MLP):
        super(eeg_encoder_mlp, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(config.freq_dim, config.hidden_dim[0]))
        self.layers.append(nn.BatchNorm1d(config.hidden_dim[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(config.dropout))

        for i in range(len(config.hidden_dim) - 1):
            self.layers.append(nn.Linear(config.hidden_dim[i], config.hidden_dim[i + 1]))
            self.layers.append(nn.BatchNorm1d(config.hidden_dim[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(config.dropout))

        self.layers.append(nn.Linear(config.hidden_dim[-1], config.out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        for layer in self.layers:
            x = layer(x)
        return x
    

class eeg_encoder_temporal(nn.Module):
    """
    Temporal Attention encoder for time series
    """
    def __init__(self, config: Config_Temporal):
        super(eeg_encoder_temporal, self).__init__()
        self.attention_layer = nn.Linear(config.time_dim, config.hidden_dim)
        self.context_vector = nn.Linear(config.hidden_dim, 1, bias=False)
        self.fc1 = nn.Linear(config.time_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(self, x):
        # 计算注意力权重
        attention_scores = self.attention_layer(x)  # [batch_size, seq_len, hidden_dim]
        attention_scores = torch.tanh(attention_scores)  # [batch_size, seq_len, hidden_dim]
        attention_scores = self.context_vector(attention_scores)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]

        # 归一化注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]

        # 计算加权和
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        weighted_sum = torch.sum(x * attention_weights, dim=1)  # [batch_size, feature_dim]

        self.attention_weights = attention_weights

        # 全连接层
        output = self.fc1(weighted_sum)
        output = F.relu(output)
        output = self.fc2(output)

        return output


class eeg_encoder_transformer(nn.Module):
    """
    Transformer for time series
    """
    def __init__(self, config: Config_Transformer):
        super(eeg_encoder_transformer, self).__init__()
        self.embedding = nn.Conv1d(config.channel, config.hidden_dim, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.fc_out = nn.Linear(config.hidden_dim * config.time_dim, config.out_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.pre_norm = config.pre_norm
        if self.pre_norm:
            self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        # 输入 x 的形状为 [batch_size, channel, embedding_dim]
        x = self.embedding(x)  # 经过 Conv1d 后，形状为 [batch_size, hidden_dim, embedding_dim]
        if self.pre_norm:
            x = self.norm(x)
        x = x.permute(2, 0, 1)  # 调整维度为 [embedding_dim, batch_size, hidden_dim]
        x = self.transformer_encoder(x)  # 经过 Transformer 编码器
        x = x.permute(1, 2, 0)  # 调整维度回到 [batch_size, hidden_dim, embedding_dim]
        x = x.reshape(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class eeg_encoder_LSTM(nn.Module):
    """
    LSTM for time series
    """
    def __init__(self, config: Config_LSTM):
        super(eeg_encoder_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=config.time_dim,
                            hidden_size=config.lstm_hidden_dim,
                            num_layers=config.lstm_layers,
                            batch_first=True,
                            dropout=config.dropout)
        self.fc = nn.Linear(config.lstm_hidden_dim, config.out_dim)
        self.dropout = nn.Dropout(config.dropout)
        # self.norm = nn.BatchNorm1d(config.out_dim)
        self.norm = nn.LayerNorm(config.out_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim]
        lstm_out = self.dropout(lstm_out)
        output = lstm_out[:, -1, :]  # final_output: [batch_size, hidden_dim]
        output = self.fc(output)  # output: [batch_size, output_dim]
        output = self.norm(output)
        return output
  

def get_eeg_encoder(config: Config):
    if config.type == "MLP":
        return eeg_encoder_mlp(config)
    elif config.type == "Temporal":
        return eeg_encoder_temporal(config)
    elif config.type == "Transformer":
        return eeg_encoder_transformer(config)
    elif config.type == "LSTM":
        return eeg_encoder_LSTM(config)
    else:
        raise NotImplementedError