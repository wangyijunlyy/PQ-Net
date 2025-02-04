import torch
from torch import nn
import torch.nn.functional as F
import math

import numpy as np
# Define a model registry
model_registry = {}

# Create a decorator to register models
def register_model(model_name):
    def decorator(cls):
        model_registry[model_name] = cls
        return cls
    return decorator

# Define a function to retrieve and instantiate the model class by model_name
def get_model_by_name(model_name, *args, **kwargs):
    model_cls = model_registry.get(model_name)
    if model_cls is None:
        raise ValueError(f"No model found with model_name{model_name}.")
    return model_cls(*args, **kwargs)

# Use the decorator to register the model class

@register_model('FANLayer')
class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias) # There is almost no difference between bias and non-bias in our experiments.
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
        self.activation = nn.GELU()        
    
    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output
    
@register_model('FANLayerGated')
class FANLayerGated(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, gated = True):
        super(FANLayerGated, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias) 
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
        self.activation = nn.GELU()        
        if gated:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
    
    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        
        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate*torch.cos(p), gate*torch.sin(p), (1-gate)*g), dim=-1)
        return output

@register_model('FAN')
class FAN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3):
        super(FAN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)   
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayer(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        # print(src.shape)
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model('FANGated')
class FANGated(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, gated = True):
        super(FANGated, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)   
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayerGated(hidden_dim, hidden_dim, gated = gated))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model('MLP')
class MLPModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, use_embedding=True):
        super(MLPModel, self).__init__()
        self.activation = nn.GELU()  
        self.layers = nn.ModuleList() 
        if use_embedding:
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        else:
            self.layers.extend([nn.Linear(input_dim, hidden_dim), self.activation])
        
        for _ in range(num_layers - 2):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src) if hasattr(self, 'embedding') else src
        for layer in self.layers:
            output = layer(output)
        return output


class RoPEPositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x


@register_model('Transformer')
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=768, num_layers=12, num_heads=12, norm_first = True, encoder_only=True, decoder_only=False):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = RoPEPositionalEncoding(hidden_dim)
        self.encoder_only = encoder_only
        self.decoder_only = decoder_only
        assert not (self.encoder_only and self.decoder_only)
        if self.encoder_only:
            encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, norm_first = norm_first)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        elif self.decoder_only:
            decoder_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, norm_first = norm_first)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        else:
            encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, norm_first = norm_first)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers//2)
            decoder_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, norm_first = norm_first)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers//2)
        self.out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, src):
        src = self.embedding(src).unsqueeze(0)
        src = self.pos_encoder(src)
        if self.encoder_only:
            src = self.transformer_encoder(src)
        elif self.decoder_only:
            src = self.transformer_decoder(src, src)
        else:
            src = self.transformer_encoder(src)
            src = self.transformer_decoder(src, src)
        output = self.out(src)
        return output
        
        
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )

        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


@register_model('KAN')
class KAN(nn.Module):
    def __init__(
        self,
        input_dim=1, 
        output_dim=1, 
        hidden_dim=128, 
        num_layers=3,
        grid_size=50,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        layers_hidden=[input_dim] + [hidden_dim] * num_layers + [output_dim]
        
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

import pennylane as qml



n_qubits = 8

dev = qml.device("default.qubit", wires=n_qubits)
# @qml.qnode(dev)
# def qnode(inputs, weights):
#     qml.AngleEmbedding(inputs, wires=range(n_qubits))
#     qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
#     qml.RX()
#     return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
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


@register_model('PQN')
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


class FourierFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, learnable_features=False):
        super(FourierFeatures, self).__init__()
        frequency_matrix = torch.normal(mean=torch.zeros(out_channels, in_channels),
                                        std=1.0)
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        if coordinates.dim() == 2:
            coordinates = coordinates.unsqueeze(0) 
        prefeatures = torch.einsum('oi,bli->blo', self.frequency_matrix.to(coordinates.device), coordinates)
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        return torch.cat((cos_features, sin_features), dim=2)

class SineLayer_bn(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, activ='relu', omega_0=30):
        super().__init__()

        self.is_first = is_first
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        if activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'sine':
            self.activ = torch.sin

    def forward(self, input):
        x1 = self.linear(input)
        x1 = self.omega_0 * self.norm(x1.permute(0, 2, 1)).permute(0, 2, 1)
        return self.activ(x1)

@register_model('Relu+Rff')
class relu_rff(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=8, num_layers=3, outermost_linear=True, activ='relu',
                 first_omega_0=30, hidden_omega_0=30, rff=True):
        super().__init__()

        self.net = []
        if rff:
            self.net.append(FourierFeatures(input_dim, hidden_dim // 2))
        else:
            self.net.append(
                SineLayer_bn(input_dim, hidden_dim, is_first=True, activ=activ, omega_0=first_omega_0))

        for i in range(num_layers):
            self.net.append(
                SineLayer_bn(hidden_dim, hidden_dim, is_first=False, activ=activ, omega_0=hidden_omega_0))

        if outermost_linear:
            self.net.append(nn.Linear(hidden_dim, output_dim))
        else:
            self.net.append(SineLayer_bn(hidden_dim, output_dim, is_first=False, activ=activ))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, idx=0):
        super().__init__()
        self.idx = idx
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = self.omega_0 * self.linear(input)
        return torch.sin(out)

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

@register_model('SIREN')
class Siren(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=256, num_layers=1, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(input_dim, hidden_dim, is_first=True, omega_0=first_omega_0, idx=1))

        for i in range(num_layers):
            self.net.append(
                SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=hidden_omega_0, idx=i + 2))

        if outermost_linear:
            final_linear = nn.Linear(hidden_dim, output_dim)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0,
                                             np.sqrt(6 / hidden_dim) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_dim, output_dim, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)
        

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output





class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
@register_model('wire')
class INR(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, 
                 num_layers=3, 
                 output_dim=1, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_dim = int(hidden_dim/np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'    
        
        # Legacy parameter
        self.pos_encode = False
            
        self.net = []
        self.net.append(self.nonlin(input_dim,
                                    hidden_dim, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(num_layers):
            self.net.append(self.nonlin(hidden_dim,
                                        hidden_dim, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_dim,
                                 output_dim,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return output.real
         
        return output
    

