import json
import matplotlib.pyplot as plt

def pad_losses_to_length(losses, target_length):
    """
    将损失数据的长度扩展到目标长度，使用最后一个值进行补充。
    新增的条目中，epoch 值逐步增加到目标长度。
    """
    current_length = len(losses)
    
    if current_length < target_length:
        last_entry = losses[-1]
        last_epoch = last_entry['epoch']
        
        # 生成补全的条目
        for i in range(1, target_length - current_length + 1):
            new_entry = {
                'epoch': last_epoch + i,
                'train_loss': last_entry['train_loss'],
                'test_loss': last_entry['test_loss']
            }
            losses.append(new_entry)
    return losses

def load_losses(file_path, max_entries=200):
    """
    加载损失日志文件，并返回一个包含最多 max_entries 条数据的列表
    """
    losses = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= max_entries:  # 只加载前 max_entries 条数据
                break
            losses.append(json.loads(line.strip()))
    return losses


def plot_loss_comparison_separate_logscale(losses_dict, save_path_train="train_loss_log_comparison.png", save_path_test="test_loss_log_comparison.png"):
    """
    分别绘制多个模型的训练损失和测试损失对比图（对数尺度），使用不同颜色和线型区分模型
    """
    linestyles = [':', '--', '-.', '-']  # 定义线型
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']  # 鲜艳的颜色
    line_width = 4  # 增加线条宽度
    grid_line_width = 1.2  # 网格线加粗

    # 绘制训练损失对比图（对数尺度）
    plt.figure(figsize=(12, 8))
    for idx, (model_name, losses) in enumerate(losses_dict.items()):
        epochs = [log['epoch'] for log in losses]
        train_losses = [log['train_loss'] for log in losses]
        plt.plot(
            epochs, train_losses, label=f"{model_name}",
            linewidth=line_width, linestyle=linestyles[idx % len(linestyles)],
            color=colors[idx % len(colors)]
        )
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("Train Loss", fontsize=20)
    plt.yscale('log')  # 设置对数尺度
    # plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", linewidth=grid_line_width, alpha=0.6)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 加粗边框
    plt.tight_layout()
    plt.savefig(save_path_train, dpi=300)  # 提高图像分辨率
    plt.close()  # 关闭当前绘图
    
    # 绘制测试损失对比图（对数尺度）
    plt.figure(figsize=(12, 8))
    for idx, (model_name, losses) in enumerate(losses_dict.items()):
        epochs = [log['epoch'] for log in losses]
        test_losses = [log['test_loss'] for log in losses]
        plt.plot(
            epochs, test_losses, label=f"{model_name}",
            linewidth=line_width, linestyle=linestyles[idx % len(linestyles)],
            color=colors[idx % len(colors)]
        )
    plt.xlabel("Epochs", fontsize=30)
    plt.ylabel("Test Loss", fontsize=30)
    plt.yscale('log')  # 设置对数尺度
    plt.tick_params(axis='both', labelsize=20)
    # plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", linewidth=grid_line_width, alpha=0.6)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 加粗边框
    plt.tight_layout()
    plt.savefig(save_path_test, dpi=300)  # 提高图像分辨率
    plt.close()  # 关闭当前绘图
def save_legend_as_image(losses_dict, save_path="legend.png"):
    """
    将图例单独保存为图片
    """
    linestyles = [':', '--', '-.', '-']  # 定义线型
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']  # 鲜艳的颜色
    plt.figure(figsize=(2, 1))
    
    # 创建一个虚拟图，用于生成图例
    for idx, model_name in enumerate(losses_dict.keys()):
        display_name = "PQ-Block" if model_name == "PQN" else model_name
        plt.plot([], [], label=f"{display_name}", linewidth=2,
                 linestyle=linestyles[idx % len(linestyles)], color=colors[idx % len(colors)])
    
    legend = plt.legend(fontsize=14, loc='center', frameon=True)
    frame = legend.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(1.2)
    
    # 仅保存图例
    plt.axis("off")  # 关闭坐标轴
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1,dpi=300)
    plt.close()

if __name__ == '__main__':
    # 假设你有多个模型的loss文件路径
    losses_dict = {}

    # 各个模型的损失文件路径
    model_paths = {
        "MLP": "./mlp_checkpoint/dataset_0/depth_2_hidden_8/losses.jsonl",
        "KAN": "./kan_checkpoint/dataset_0/losses_grid_50.jsonl",
        "Transformer": "./transformer_checkpoint/dataset_0/depth_2_hidden_8/losses.jsonl",
        "PQN": "./PQN_checkpoint/dataset_0/depth_2_hidden_8/losses.jsonl"
    }
    
    target_length = 2000
    max_entries = 200  # 只加载前200条数据
    
    # 加载每个模型的损失数据
    for model_name, file_path in model_paths.items():
        losses = load_losses(file_path, max_entries=max_entries)
        # if model_name == "PQN":
        #     losses = pad_losses_to_length(losses, target_length)
        #     print(len(losses))
        losses_dict[model_name] = losses
    
    # 分别绘制训练和测试损失对比图（对数尺度）
    plot_loss_comparison_separate_logscale(
        losses_dict,
        save_path_train="train_loss_log_comparison_0.png",
        save_path_test="test_loss_log_comparison_0.png"
    )
    save_legend_as_image(losses_dict, save_path="legend.png")