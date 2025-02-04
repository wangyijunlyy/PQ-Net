import torch, os, argparse, json
import torch.nn as nn
from tqdm import tqdm
from kan import LBFGS
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP')
    parser.add_argument('--dataset_idx', type=int, default=0, help='Dataset index')
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--save_dir', type=str, default='kan_checkpoint')
    return parser.parse_args()


def load_dataset(args, dataset_idx):
    print(f'Loading dataset_{dataset_idx} from {args.dataset_dir}/dataset_{dataset_idx}.pt')

    dataset = torch.load(f'{args.dataset_dir}/dataset_{dataset_idx}.pt')
    dataset['train_input'] = dataset['train_input'].to(device)
    dataset['test_input'] = dataset['test_input'].to(device)
    dataset['train_label'] = dataset['train_label'].to(device)
    dataset['test_label'] = dataset['test_label'].to(device)
    return dataset


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth

        layers = []
        
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_param_size(self):
        total_parameters = 0
        total_parameters += (self.input_size + 1) * self.hidden_size
        for _ in range(self.depth - 1):
            total_parameters += (self.hidden_size + 1) * self.hidden_size
        total_parameters += (self.hidden_size + 1) * self.output_size
        return total_parameters


def train_with_test(model, dataset, ckpt_dir, log_file):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    # Criterion and Optimizer
    criterion = nn.MSELoss()
    optimizer = LBFGS(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.01,
        history_size=40, 
        line_search_fn="strong_wolfe", 
        tolerance_grad=1e-32, 
        tolerance_change=1e-32, 
        tolerance_ys=1e-32
    )
    
    # Ensure log file directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Storage for logging loss values
    losses = []

    model.train()
    num_epochs = 2000

    for epoch in tqdm(range(num_epochs)):
        def closure():
            optimizer.zero_grad()
            output = model(dataset['train_input'])
            train_loss = criterion(output, dataset['train_label'])
            train_loss.backward()
            return train_loss
        
        # Perform optimization step
        optimizer.step(closure)

        # Log training loss
        with torch.no_grad():
            train_loss = criterion(model(dataset['train_input']), dataset['train_label']).item()
            test_loss = criterion(model(dataset['test_input']), dataset['test_label']).item()
            losses.append({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss})
        
        # Save loss to log file
        with open(log_file, 'a') as f:
            f.write(json.dumps(losses[-1]) + '\n')

        # Optional: Print epoch loss
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
    
    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))
    return test_loss,losses


if __name__ == '__main__':
    args = parse_args()
    if args.dataset_idx == 0:
        dataset = load_dataset(args, 0)
        input_size, output_size = 1, 1
    elif args.dataset_idx == 1:
        dataset = load_dataset(args, 1)
        input_size, output_size = 2, 1
    elif args.dataset_idx == 2:
        dataset = load_dataset(args, 2)
        input_size, output_size = 2, 1
    elif args.dataset_idx == 3:
        dataset = load_dataset(args, 3)
        input_size, output_size = 4, 1
    
    save_dir = f'{args.save_dir}/dataset_{args.dataset_idx}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open(f'{save_dir}/results.jsonl', 'w')

    def plot_loss_curve(losses, save_path="loss_curve.png"):
        """
        Plot train and test loss over epochs.
        """
        epochs = [log['epoch'] for log in losses]
        train_losses = [log['train_loss'] for log in losses]
        test_losses = [log['test_loss'] for log in losses]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Train Loss", color='blue', linewidth=2)
        plt.plot(epochs, test_losses, label="Test Loss", color='red', linewidth=2)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Train and Test Loss Over Epochs", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(save_path)
        
    for depth in [2]:
        for hidden_size in [8]:
            print(f'Depth: {depth}, Hidden size: {hidden_size}')
            model = MLP(input_size, hidden_size, output_size,depth).to(device)
            param_size = sum(p.numel() for p in model.parameters())
            ckpt_dir = f'{save_dir}/depth_{depth}_hidden_{hidden_size}'
            test_loss, losses = train_with_test(model, dataset, ckpt_dir,log_file=f"{ckpt_dir}/losses.jsonl")
            plot_loss_curve(losses, save_path=f"{ckpt_dir}/loss_curve.png")
            output_js = {}
            output_js['depth'] = depth
            output_js['hidden_size'] = hidden_size
            # param_size = model.get_param_size()
            output_js['param_size'] = param_size
            output_js['test_loss'] = test_loss
            log_file.write(json.dumps(output_js) + '\n')
            log_file.flush()
    log_file.close()



