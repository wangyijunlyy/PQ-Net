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


class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
        self.activation = nn.GELU()        
    
    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output

class FAN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3):
        super(FAN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)   
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayer(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

import pennylane as qml



n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qnode(inputs, weights, rx_angles):
    # 嵌入输入数据
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # 添加纠缠层
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    # 对每个量子比特应用 RX 门
    for i in range(n_qubits):
        qml.RX(rx_angles[i], wires=i)
    # 测量 PauliZ 期望值
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


class PQN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=8, num_layers=3):
        super(PQN, self).__init__()
        self.clayer_1 = nn.Linear(input_dim, hidden_dim)  
        self.clayer_2 = torch.nn.Linear(hidden_dim, output_dim) 
        
        # weight_shapes = {"weights": (n_layers, n_qubits)}
        weight_shapes = {
    "weights": (num_layers, n_qubits),  # BasicEntanglerLayers 的权重
    "rx_angles": (n_qubits,)         # 每个量子比特的 RX 门参数
}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)   

    def forward(self, src):
        # print(src.shape)
        layers = [self.clayer_1, self.qlayer, self.clayer_2]
        model = torch.nn.Sequential(*layers)
        
        output = model(src)
        return output





# def train_with_test(model, dataset, ckpt_dir):

#     if not os.path.exists(ckpt_dir):
#         os.makedirs(ckpt_dir)
#     criterion = nn.MSELoss()
#     optimizer = LBFGS(filter(lambda p: p.requires_grad, model.parameters()), 
#                       lr=0.0001,
#                       history_size=40, 
#                       line_search_fn="strong_wolfe", 
#                       tolerance_grad=1e-32, 
#                       tolerance_change=1e-32, 
#                       tolerance_ys=1e-32)
    
#     model.train()
#     for _ in tqdm(range(1800)):
#         def closure():
#             optimizer.zero_grad()
#             output = model(dataset['train_input'])
#             loss = criterion(output, dataset['train_label'])
#             loss.backward()
#             return loss
#         optimizer.step(closure)
    
#     torch.save(model.state_dict(), f'{ckpt_dir}/model.pth')

#     model.eval()
#     with torch.no_grad():
#         output = model(dataset['test_input'])
#         test_loss = criterion(output, dataset['test_label']).item()
#     return test_loss

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
    num_epochs = 200

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
        
    for depth in [1]:
        for hidden_size in [8]:
            print(f'Depth: {depth}, Hidden size: {hidden_size}')
            model = PQN(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size, num_layers=depth).to(device)
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
