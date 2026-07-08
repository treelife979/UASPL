import torch.nn as nn
import torch
from torch import optim

# SimpleMLP
class Net(nn.Module):
    def __init__(self, input_size,  hidden_size, num_classes, dropout_rate=0.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU()  # 比ReLU更平滑，收敛更好

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

def initial_model(X_train, y_train, lr, dropout_rate=0.0):
    # 初始化模型、优化器
    input_size = X_train.shape[1]
    hidden_size = 128
    num_classes = int(torch.max(y_train).item()) + 1

    model = Net(input_size, hidden_size, num_classes, dropout_rate)
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)

    return model, optimizer