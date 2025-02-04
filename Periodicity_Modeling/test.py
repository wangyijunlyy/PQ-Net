# train.py
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import argparse
from generate_periodic_data import gen_periodic_data, plot_periodic_data, plot_truth_data
import logging
from datetime import datetime
sns.set_style('whitegrid')

def save_loss_dict(loss_dict, path, model_name, periodic_type):
    """保存 loss 字典到 JSON 文件"""
    save_path = os.path.join(path, 'loss_records')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    file_path = os.path.join(save_path, f'loss_{model_name}_{periodic_type}.json')
    with open(file_path, 'w') as f:
        json.dump(loss_dict, f)
    return file_path

def load_loss_dict(path, model_name, periodic_type):
    """从 JSON 文件加载 loss 字典"""
    file_path = os.path.join(path, 'loss_records', f'loss_{model_name}_{periodic_type}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

model_names = ['FAN', 'FANGated', 'MLP', 'KAN', 'Transformer', 'QAN','Relu+Rff', 'SIREN', 'wire', 'FNO']
periodic_types = ['sin','d2_s1', 'd5_s1','d6_s1']

parser = argparse.ArgumentParser()
parser.add_argument('--periodic_type', type=str, choices=periodic_types, help='periodic type', default='sin')
parser.add_argument('--path', type=str, help='path')
parser.add_argument('--model_name', type=str, choices=model_names, help='model name', default='FAN')

args = parser.parse_args()

# Set up logging
if not os.path.exists(f'{args.path}/logs'):
    os.makedirs(f'{args.path}/logs')

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'{args.path}/logs/{args.model_name}_{args.periodic_type}_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # This will also print to console
    ]
)

# Log initial configuration
logging.info(f"Starting training with configuration:")
logging.info(f"Model: {args.model_name}")
logging.info(f"Periodic Type: {args.periodic_type}")
logging.info(f"Output Path: {args.path}")

t, data, t_test, data_test, PERIOD, BATCHSIZE, NUMEPOCH, PRINTEPOCH, lr, wd, y_uper, y_lower = gen_periodic_data(args.periodic_type)
# plot_truth_data(t, data, t_test, data_test,args.path, y_uper, y_lower)
# Log hyperparameters
logging.info(f"Hyperparameters:")
logging.info(f"Batch Size: {BATCHSIZE}")
logging.info(f"Number of Epochs: {NUMEPOCH}")
logging.info(f"Learning Rate: {lr}")
logging.info(f"Weight Decay: {wd}")

# Check and create the output directory
if not os.path.exists(f'{args.path}'):
    os.makedirs(f'{args.path}')

# Prepare dataset and dataloaders
t_tensor = torch.tensor(t).float().unsqueeze(1)  
data_tensor = torch.tensor(data).float().unsqueeze(1)  
dataset = TensorDataset(t_tensor, data_tensor)

dataloader_train = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load model
from architecture import get_model_by_name

model = get_model_by_name(args.model_name, input_dim=1, output_dim=1, num_layers=3).to(device)
logging.info(f"Model architecture:\n{str(model)}")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

t_test_tensor = torch.tensor(t_test).float().unsqueeze(1)  
data_test_tensor = torch.tensor(data_test).float().unsqueeze(1) 
dataset_test = TensorDataset(t_test_tensor, data_test_tensor)

dataloader_test = DataLoader(dataset_test, batch_size=BATCHSIZE)

# 创建记录 loss 的字典
loss_record = {
    'train_losses': [],
    'test_losses': [],
    'final_test_loss': None,
    'learning_rates': [],
    'hyperparameters': {
        'batch_size': BATCHSIZE,
        'num_epochs': NUMEPOCH,
        'learning_rate': lr,
        'weight_decay': wd
    }
}

# Train the model
num_epochs = NUMEPOCH
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
    loss_record['learning_rates'].append(current_lr)
    for x, y in dataloader_train:
       
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.squeeze(0), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(dataloader_train)
    loss_record['train_losses'].append(avg_train_loss)
    
    # Test the model
    model.eval()
    total_test_loss = 0
    result = []
    with torch.no_grad():
        for x, y in dataloader_test:
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)
            result.extend(predictions.cpu().squeeze())
            test_loss = criterion(predictions.squeeze(0), y)
            total_test_loss += test_loss.item()
    
    avg_test_loss = total_test_loss / len(dataloader_test)
    loss_record['test_losses'].append(avg_test_loss)
    # scheduler.step()
    scheduler.step(avg_test_loss)
    logging.info(f'Epoch {epoch}:')
    logging.info(f'  Train Loss: {avg_train_loss:.6f}')
    logging.info(f'  Test Loss: {avg_test_loss:.6f}')
    logging.info(f'  current lr: {current_lr:.6f}')

    if epoch % PRINTEPOCH == 0:
        plot_periodic_data(t, data, t_test, data_test, result, args, epoch, args.path, y_uper, y_lower)
        logging.info(f'  Saved plot for epoch {epoch}')

# Save the final test loss
loss_record['final_test_loss'] = avg_test_loss

# Save the loss record
save_path = save_loss_dict(loss_record, args.path, args.model_name, args.periodic_type)
logging.info(f'Loss record saved to {save_path}')

# Save the model
model_save_path = f'{args.path}/{args.model_name}.pth'
torch.save(model.state_dict(), model_save_path)
logging.info(f'Model saved to {model_save_path}')

# Plot the current model's loss curves
plt.figure(figsize=(10, 6))
plt.plot(loss_record['train_losses'], label='Train Loss', color='blue')
plt.plot(loss_record['test_losses'], label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Test Loss for {args.model_name}')
plt.legend()
plt.grid(True)

loss_plot_path = f'{args.path}/loss_plot_{args.model_name}.png'
plt.savefig(loss_plot_path)
logging.info(f'Loss plot saved to {loss_plot_path}')