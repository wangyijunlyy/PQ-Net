import torch
import argparse, json
import numpy as np
import os
from kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Train KAN')
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

def compute_kan_size(width, grid, k):
    kan_size = 0
    for i in range(len(width) - 1):
        kan_size += (width[i][0] * width[i+1][0] * (grid + k + 3) + width[i+1][0])
    return kan_size

def numpy_to_python(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.item()
    return obj

if __name__ == '__main__':
    args = parse_args()
    if args.dataset_idx == 0:
        dataset = load_dataset(args, 0)
        width = [1, 1]
    elif args.dataset_idx == 1:
        dataset = load_dataset(args, 1)
        width = [2, 1, 1]
    elif args.dataset_idx == 2:
        dataset = load_dataset(args, 2)
        width = [2, 2, 1]
    elif args.dataset_idx == 3:
        dataset = load_dataset(args, 3)
        width = [4, 4, 2, 1]
    else:
        raise ValueError('Invalid dataset index')
    
    save_dir = f'{args.save_dir}/dataset_{args.dataset_idx}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = open(f'{save_dir}/results.jsonl', 'a')
    
    grids = [3]
    for i, grid in enumerate(grids):
        if i == 0:
            ckpt_dir = f'{save_dir}/ckpt'
            model = KAN(width=width, grid=grid, k=3, device=device, ckpt_path=ckpt_dir)
        else:
            model = model.refine(grid)
        
        results = model.fit(dataset, opt="LBFGS", steps=1800, lr=0.01)
        
        # 保存每个grid的训练过程
        output_js = {
            'grid': grid,
            'param_size': compute_kan_size(width, grid, 3),
            'train_loss': numpy_to_python(results['train_loss'][-1]),
            'test_loss': numpy_to_python(results['test_loss'][-1])
        }
        
        log_file.write(json.dumps(output_js) + '\n')
        log_file.flush()

        # 保存每个grid的完整训练过程
        losses_file = f'{save_dir}/losses_grid_{grid}.jsonl'
        os.makedirs(os.path.dirname(losses_file), exist_ok=True)
        losses = []
        for epoch, (train_loss, test_loss) in enumerate(zip(results['train_loss'], results['test_loss'])):
            loss_entry = {
                "epoch": epoch + 1,
                "train_loss": numpy_to_python(train_loss),
                "test_loss": numpy_to_python(test_loss)
            }
            losses.append(loss_entry)
            
            with open(losses_file, 'a') as f:
                f.write(json.dumps(loss_entry) + '\n')

    log_file.close()