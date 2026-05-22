# Developing a Neural Network Regression Model 

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
A company has collected a dataset containing various input features and corresponding numerical output values related to a specific problem (such as sales, price, or demand prediction). The relationship between the input variables and output is complex and cannot be accurately modeled using simple statistical methods.

To solve this problem, the company wants to develop a neural network-based regression model that can learn patterns from the existing data. The model should be trained using historical data so that it can understand how input features influence the output values.

Once trained, the model will be used to predict continuous numerical outputs for new, unseen data points. This will help the company make better decisions based on accurate predictions.

The goal is to minimize prediction error and improve the model’s performance using appropriate training techniques such as backpropagation and optimization algorithms

## Neural Network Model
<img width="1126" height="856" alt="585527108-0b9368da-34ee-4f20-aa49-5de57c9e1793" src="https://github.com/user-attachments/assets/0ef3745c-20fa-4d6d-9a19-5df9b20d9fd4" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

### Name:MAGESHWARAN T.A

### Register Number: 212224230146

## PROGRAM 
```
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/drive/MyDrive/DL EXP-1 - Sheet1.csv')
X = dataset1[['input']].values
y = dataset1[['output']].values

dataset1.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: MAGESHWARAN T.A
# Register Number: 212224230146
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)

# Name: AJITH A
# Register Number: 212224230012
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
# train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
# This cell was commented out to avoid the NameError until the function is defined.

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

### Dataset Information
<img width="372" height="506" alt="image" src="https://github.com/user-attachments/assets/7f107b75-ff71-418c-897b-afe4b0c2cfea" />




### OUTPUT
<img width="953" height="240" alt="image" src="https://github.com/user-attachments/assets/120710d6-61bb-464a-a92b-9c3e7bd3f6eb" />




### Training Loss Vs Iteration Plot
<img width="808" height="575" alt="image" src="https://github.com/user-attachments/assets/4197f566-ea43-4877-95dc-85390266163c" />



### New Sample Data Prediction

<img width="307" height="42" alt="image" src="https://github.com/user-attachments/assets/2128f2b8-f776-49a0-95a5-88e039520764" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.


