class Config:
    def __init__(self):
        self.data_len = 501  # 原始时域信号长度
        self.time_cut = [40, 440]  # 时域信号截取范围
        self.time_dim = 400
        self.freq_dim = 310
        self.channel = 62
        self.out_dim = 768  # 输出维度
        self.train_ratio = 0.6  # 训练集比例

    def load_config(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class Config_MLP(Config):
    def __init__(self):
        super().__init__()
        self.type = "MLP"
        self.lr = 0.01  # 学习率
        self.min_lr = 0.0  # 最小学习率
        self.temperature = 0.1  # InfoNCE温度
        self.weight_decay = 0.001  # 权重衰减
        self.num_epoch = 500  # 训练轮数
        self.warmup_epochs = 50  # 预热轮数
        self.batch_size = 128  # 批次大小
        self.clip_grad = 0.8  # 梯度裁剪
        self.hidden_dim = [512, 512]  # 隐藏维度
        self.dropout = 0.5  # 丢弃率


class Config_Temporal(Config):
    def __init__(self):
        super().__init__()
        self.type = "Temporal"
        self.lr = 0.0001  # 学习率
        self.min_lr = 0.0  # 最小学习率
        self.temperature = 0.03  # InfoNCE温度
        self.weight_decay = 0.001  # 权重衰减
        self.num_epoch = 500  # 训练轮数
        self.warmup_epochs = 50  # 预热轮数
        self.batch_size = 128  # 批次大小
        self.clip_grad = 0.8  # 梯度裁剪
        self.hidden_dim = 1024  # 隐藏维度
        self.dropout = 0.5  # 丢弃率


class Config_Transformer(Config):
    def __init__(self):
        super().__init__()
        self.type = "Transformer"
        self.lr = 0.0000001  # 学习率
        self.min_lr = 0.0  # 最小学习率
        self.temperature = 0.1
        self.weight_decay = 0.05  # 权重衰减
        self.num_epoch = 30  # 训练轮数
        self.warmup_epochs = 10  # 预热轮数
        self.batch_size = 32  # 批次大小
        self.clip_grad = 0.8  # 梯度裁剪
        self.depth = 2  # 深度
        self.hidden_dim = 768  # 隐藏维度
        self.nhead = 8  # 头数
        self.dropout = 0.1  # 丢弃率
        self.dim_feedforward = 768  # 前馈网络维度
        self.activation = "relu"  # 激活函数
        self.pre_norm = False

class Config_LSTM(Config):
    def __init__(self):
        super().__init__()
        self.type = "LSTM"
        self.lr = 0.00015  # 学习率
        self.min_lr = 0.0  # 最小学习率
        self.weight_decay = 0.05  # 权重衰减
        self.num_epoch = 500  # 训练轮数
        self.warmup_epochs = 10  # 预热轮数
        self.temperature = 0.05
        self.batch_size = 128  # 批次大小
        self.clip_grad = 0.8  # 梯度裁剪
        self.lstm_hidden_dim = 1024
        self.lstm_layers = 2
        self.dropout = 0.1
        self.fc_dim = 1024
