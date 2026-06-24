import random
import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


dataset = fetch_california_housing(as_frame=True)
X = dataset.data
Y = dataset.target

# print(X.head())
# print(X.info())
# print(X.describe())
# print(Y.describe())


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, Y_train.shape)


batch_size = 128
X_train_tensor = torch.FloatTensor(np.array(X_train))
Y_train_tensor = torch.FloatTensor(np.array(Y_train))

X_test_tensor = torch.FloatTensor(np.array(X_test))
Y_test_tensor = torch.FloatTensor(np.array(Y_test))


train_data = TensorDataset(X_train_tensor, Y_train_tensor)
test_data = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last = True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last = True)

X_batch, Y_batch = next(iter(train_loader))


D_in, H, D_out = X_batch.shape[1], 64, 1

class MyModule(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Linear(H,H),
        nn.ReLU(),
        nn.Linear(H, D_out)
    )

  def forward(self, X):
    return self.net(X)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())
model = MyModule()
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)



import matplotlib.pyplot as plt
epochs = 3
history = []

for epoch in range(epochs):
  epoch_losses = []
  for x_batch, y_batch in train_loader:

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)


    y_pred = model(x_batch)

    loss = mse_loss(y_pred.view(-1), y_batch.view(-1))
    epoch_losses.append(loss.item())

    optim.zero_grad()

    loss.backward()

    optim.step()

  epoch_mean_loss = sum(epoch_losses) / len(epoch_losses)
  history.append(epoch_mean_loss)

  print(f"epoch {epoch+1}: loss: {history[-1]}")

plt.plot(history)
plt.show()



model.eval()

test_losses = []
all_preds = []
all_targets = []


with torch.no_grad():
    for x_batch, y_batch in test_data:
        x_batch = x_batch.to(device).float()
        y_batch = y_batch.to(device).float()


        y_pred = model(x_batch)

        loss = mse_loss(y_pred.view(-1), y_batch.view(-1))
        test_losses.append(loss.item())


        all_preds.extend(y_pred.view(-1).cpu().numpy())
        all_targets.extend(y_batch.view(-1).cpu().numpy())


mean_test_loss = sum(test_losses) / len(test_losses)
print(f"Среднеквадратичная ошибка (MSE) на тестовых данных: {mean_test_loss:.4f}")
