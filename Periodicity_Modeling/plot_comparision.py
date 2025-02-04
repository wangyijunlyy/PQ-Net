# plot_comparison.py
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_loss_dict(path, model_name, periodic_type):
    """从 JSON 文件加载 loss 字典"""
    file_path = os.path.join(path, 'loss_records', f'loss_{model_name}_{periodic_type}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def plot_multiple_losses(loss_records, save_path, periodic_type):
    """绘制多个模型的 loss 对比图"""
    plt.figure(figsize=(12, 8))
    
    for model_name, losses in loss_records.items():
        plt.plot(losses['test_losses'], label=f'{model_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    # plt.title(f'Test Loss Comparison ({periodic_type})')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.yscale('log')  # 使用对数尺度更容易看出差异
    
    save_file = os.path.join(save_path, f'model_comparison_{periodic_type}.png')
    plt.savefig(save_file)
    plt.close()
    print(f'Comparison plot saved to {save_file}')
    
    # 打印最终测试 loss
    print("\nFinal Test Losses:")
    for model_name, losses in loss_records.items():
        print(f"{model_name}: {losses['final_test_loss']:.6f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.', help='Path to the results directory')
    parser.add_argument('--periodic_type', type=str, default='d5_s1', help='Periodic type to compare')
    args = parser.parse_args()
    
    model_names = ['MLP', 'KAN', 'Transformer',  'Relu+Rff', 'SIREN', 'QAN']
    
    # 加载所有可用的模型结果
    loss_records = {}
    for model_name in model_names:
        loss_data = load_loss_dict(f'{args.periodic_type}_{model_name}', model_name, args.periodic_type)
        if loss_data is not None:
            loss_records[model_name] = loss_data
    
    if len(loss_records) > 0:
        plot_multiple_losses(loss_records, args.path, args.periodic_type)
    else:
        print("No loss records found!")

if __name__ == '__main__':
    main()