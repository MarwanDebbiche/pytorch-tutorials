import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import Net
import pandas as pd
import numpy as np
from datetime import datetime

train = pd.read_csv("../data/digit-recognizer/train.csv")
test = pd.read_csv("../data/digit-recognizer/test.csv")

X_train = train.drop(columns="label").values.reshape(-1, 1, 28, 28) / 255
y_train = train["label"].values
X_test = test.values.reshape(-1, 1, 28, 28) / 255

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
EPOCHS = 3

log_dir = f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
print(f"Tensorboard log directory : {log_dir}")
writer = SummaryWriter(log_dir=log_dir)

inputs = torch.from_numpy(X_train).type(torch.FloatTensor)
labels = torch.from_numpy(y_train)

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()

    optimizer.step()

    epoch_loss = loss.item()

    writer.add_scalar('loss', epoch_loss, epoch)
    print("epoch : ", epoch + 1, "loss : ", epoch_loss)

with torch.no_grad():
    inputs = torch.from_numpy(X_test).type(torch.FloatTensor)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted_np = predicted.numpy()

    print(predicted_np[:20])

# PATH = './digit.pth'
# torch.save(net.state_dict(), PATH)
