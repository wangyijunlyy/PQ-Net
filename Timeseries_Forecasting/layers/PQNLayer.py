from torch import  nn
import torch
import pennylane as qml


class PQN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=8, num_layers=3):
        super(PQN, self).__init__()
        n_qubits = hidden_dim
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights, rx_angles):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            # for i in range(n_qubits):
            #     qml.CNOT(wires=[i, (i + 1) % n_qubits])  # 创建邻近比特之间的纠缠
            
            # for i in range(n_qubits):
            #     qml.CNOT(wires=[i, (i + 1) % n_qubits])  # 创建邻近比特之间的纠缠
           
            # for i in range(n_qubits):
            #     qml.CNOT(wires=[i, (i + 1) % n_qubits])  # 创建邻近比特之间的纠缠
            
            for i in range(n_qubits):
                qml.RX(rx_angles[i], wires=i)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        self.clayer_1 = nn.Linear(input_dim, hidden_dim)  
        self.clayer_2 = torch.nn.Linear(hidden_dim, output_dim) 
       
        weight_shapes = {
    "weights": (num_layers, n_qubits),  # BasicEntanglerLayers 的权重
    "rx_angles": (n_qubits,)         # 每个量子比特的 RX 门参数
}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)   

    def forward(self, src):
        # print(src.shape)
        layers = [self.clayer_1, self.qlayer, self.clayer_2]
        # x1 = self.clayer_1(src)
        # print(x1.device)
        # x2 = self.qlayer(x1)
        # print(x2.device)
        # output = self.clayer_2(x2)
        model = torch.nn.Sequential(*layers).to(src.device)
        
        output = model(src)
        
        return output