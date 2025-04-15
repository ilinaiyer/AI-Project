import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
class SimpleClassifier(nn.Module):
   def __init__(self, n_features, n_classes):
       super(SimpleClassifier, self).__init__()
       self.fc1 = nn.Linear(n_features, 128)  # First fully connected layer
       self.bn1 = nn.BatchNorm1d(128)
       self.fc2 = nn.Linear(128, 64)         # Second fully connected layer
       self.bn2 = nn.BatchNorm1d(64)
       self.fc3 = nn.Linear(64, n_classes)   # Output layer
       self.relu = nn.ReLU()                  # Activation function
       self.dropout = nn.Dropout(0.3)


   def forward(self, x):
       x = self.bn1(self.relu(self.fc1(x)))
       x = self.dropout(x)
       x = self.bn2(self.relu(self.fc2(x)))
       x = self.fc3(x)  # No activation function here, as we'll use CrossEntropyLoss
       return x
def columnToList(column_name):
   arr = []
   for i in range(len(df.loc[:, column_name])):
       arr.append((df.loc[:, column_name][i]))
   return arr
df = pd.read_csv("careermap.csv")
xx = []
for i in range(len(df.columns) - 1):
   xx.append(columnToList(df.columns[i]))
x = np.array(xx)
x = x.T


le = sklearn.preprocessing.LabelEncoder()
Career = df.loc[:, "Role"]
le.fit(Career)
Career = le.transform(Career)


X_train, X_test, y_train, y_test = train_test_split(x, Career, test_size=0.5, random_state=0)
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()


torch.manual_seed(0)
model = SimpleClassifier(x.shape[1], len(le.classes_)).float()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


train_acc = []
test_acc = []


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.3)
for epoch in range(200):
   model.train()
   optimizer.zero_grad()


   outputs = model(X_train)
   loss = loss_fn(outputs, y_train)
   loss.backward()
   optimizer.step()
   scheduler.step()




   model.eval()
   with torch.no_grad():
       train_preds = model(X_train)
       train_pred_labels = torch.argmax(train_preds,dim=1)
       train_correct = (train_pred_labels == y_train).sum().item()
       train_acc.append(train_correct / len(y_train))


       test_preds = model(X_test)
       test_pred_labels = torch.argmax(test_preds, dim=1)
       test_correct = (test_pred_labels == y_test).sum().item()
       test_acc.append(test_correct / len(y_test))

      
with torch.no_grad():
   test_pred = model(X_test)
   test_pred_labels = torch.argmax(test_pred, dim=1)
   unique_classes, counts = torch.unique(test_pred_labels, return_counts=True)
   print("Unique predicted classes and counts:", unique_classes.numpy(), counts.numpy())




plt.plot(train_acc, label="Train Accuracy", color="blue")
plt.plot(test_acc, label="Test Accuracy", color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
