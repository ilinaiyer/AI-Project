import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers

class SimpleClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)         # Second fully connected layer
        self.fc3 = nn.Linear(64, n_classes)   # Output layer
        self.relu = nn.ReLU()                  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function here, as we'll use CrossEntropyLoss
        return x
def columnToList(column_name):
    arr = []
    for i in range(len(df.loc[:, column_name])):
        arr.append((df.loc[:, column_name][i]))
    return arr
print("hi")
df = pd.read_csv("Data_final.csv")
O = columnToList("O_score")
C = columnToList("C_score")
E = columnToList("E_score")
A = columnToList("A_score")
N = columnToList("N_score")
Numerical = columnToList("Numerical Aptitude")
Spatial = columnToList("Spatial Aptitude")
Perceptual = columnToList("Perceptual Aptitude")
Abstract = columnToList("Abstract Reasoning")
Verbal = columnToList("Verbal Reasoning")
x = np.stack((O, C, E, A, N, Numerical, Spatial, Perceptual, Abstract, Verbal))
x = x.T
le = sklearn.preprocessing.LabelEncoder()
Career = df.loc[:, "Career"]
le.fit(Career)
Career = le.transform(Career)
X_train, X_test, y_train, y_test = train_test_split(x, Career, test_size=0.2, random_state=0)
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()
torch.manual_seed(0)
model = models.Sequential([
    layers.Input(shape=(256,256,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    #layers.Conv2D(64, (3,2), activation='relu'),
    #layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_loss = []
test_loss = []
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.3)
for epoch in range(500):
    optimizer.zero_grad()
    y_pred = model.forward(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    print(loss)
    train_loss.append(loss.detach().numpy())
    scheduler.step()
    with torch.no_grad():
        testloss = loss_fn(model.forward(X_test), y_test)
        test_loss.append(testloss.numpy())
plt.plot(train_loss, color = "blue")
plt.plot(test_loss, color = "red")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.show()