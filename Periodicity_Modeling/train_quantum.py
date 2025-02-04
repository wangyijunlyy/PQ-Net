import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import TensorDataset, DataLoader
import argparse
from generate_periodic_data import gen_periodic_data, plot_periodic_data
sns.set_style('whitegrid')

model_names = ['FAN', 'FANGated', 'MLP', 'KAN', 'Transformer', 'QAN']
periodic_types = ['sin', 'mod', 'complex_1', 'complex_2', 'complex_3', 'complex_4', 'complex_5', 'complex_6']

parser = argparse.ArgumentParser()
parser.add_argument('--periodic_type', type=str, choices=periodic_types, help='periodic type', default='sin')
parser.add_argument('--path', type=str, help='path')
parser.add_argument('--model_name', type=str, choices=model_names, help='model name', default='FAN')

args = parser.parse_args()

t, data, t_test, data_test, PERIOD, BATCHSIZE, NUMEPOCH, PRINTEPOCH, lr, wd, y_uper, y_lower = gen_periodic_data(args.periodic_type)

# Check and create the output directory
path = args.path
if not os.path.exists(f'{path}'):
    os.makedirs(f'{path}')

# Prepare dataset and dataloaders
t_tensor = torch.tensor(t,dtype=torch.float32).unsqueeze(1)  
data_tensor = torch.tensor(data,dtype=torch.float32).unsqueeze(1)  
dataset = TensorDataset(t_tensor, data_tensor)

dataloader_train = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
from architecture import get_model_by_name

print(f'model name: {args.model_name}')
model = get_model_by_name(args.model_name, input_dim=1, output_dim=1, num_layers=3).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

t_test_tensor = torch.tensor(t_test,dtype=torch.float32).unsqueeze(1)  
data_test_tensor = torch.tensor(data_test,dtype=torch.float32).unsqueeze(1) 
dataset_test = TensorDataset(t_test_tensor, data_test_tensor)

dataloader_test = DataLoader(dataset_test, batch_size=BATCHSIZE)

# List to store losses for plotting later
train_losses = []
test_losses = []

# Train the model
num_epochs = NUMEPOCH
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, y in dataloader_train:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.squeeze(0), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Save training loss for plotting
    train_losses.append(total_loss / len(dataloader_train))
    print(f'Epoch {epoch}, Train Loss {total_loss / len(dataloader_train)}')
    model.eval()

    result = []
    # Test the model
    total_test_loss = 0
    with torch.no_grad():
        for x, y in dataloader_test:
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)
            result.extend(predictions.cpu().squeeze())
            test_loss = criterion(predictions.squeeze(0), y)
            total_test_loss += test_loss.item()
    
    # Save test loss for plotting
    test_losses.append(total_test_loss / len(dataloader_test))

    print(f'Epoch {epoch}, Test Loss {total_test_loss / len(dataloader_test)}')
    if epoch % PRINTEPOCH == 0:
    # Plot periodic data
        plot_periodic_data(t, data, t_test, data_test, result, args, epoch, path, y_uper, y_lower)

# Save the model after training
torch.save(model.state_dict(), f'{args.model_name}.pth')

# Final evaluation
model.eval()
total_test_loss = 0
with torch.no_grad():
    for x, y in dataloader_test:
        x = x.to(device)
        y = y.to(device)
        predictions = model(x)
        test_loss = criterion(predictions.squeeze(0), y)
        total_test_loss += test_loss.item()
    print(f'Final Epoch, Test Loss {total_test_loss / len(dataloader_test)}')

# Plot train and test losses
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_losses, label='Train Loss', color='blue')
plt.plot(range(num_epochs), test_losses, label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss per Epoch')
plt.legend()
plt.grid(True)

# Save the loss plot
plt.savefig(f'{path}/loss_plot.png')
