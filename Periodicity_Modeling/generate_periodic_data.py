from tkinter import OFF
import numpy as np
np.random.seed(42)
def sawtooth_wave(t, n):
    """Generate a single term of the sawtooth wave harmonic series."""
    return (t / np.pi) - np.floor(t / np.pi + 0.5)

def gen_periodic_data(periodic_type):

    if periodic_type == 'd2_s1':
        def generate_periodic_data(num_samples, num_periods=100, is_train = True):
            if is_train:
                t = np.linspace(-num_periods * np.pi, num_periods * np.pi, num_samples)
            else:
                t = np.linspace(-num_periods * 3 * np.pi, num_periods * 3 * np.pi, num_samples)
            data = np.sin(t) 
            return t, data
        degree = 2  # degree of the target function
        scaling = 1  # scaling of the data
        coeffs = [0.15 + 0.15j] * degree  # coefficients of non-zero frequencies
        coeff0 = 0.1  # coefficient of zero frequency


        def target_function(x):
            """Generate a truncated Fourier series, where the data gets re-scaled."""
            
            res = coeff0
            for idx, coeff in enumerate(coeffs):
                exponent = np.complex128(scaling * (idx + 1) * x * 1j)
                conj_coeff = np.conjugate(coeff)
                res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
            return np.real(res)
        
        print(f'generate data from the {periodic_type} function')

        PERIOD = 1
        BATCHSIZE = 16
        NUMEPOCH = 501
        PRINTEPOCH = 100
        lr = 1e-1 # FAN调小:e-5,QAN，siren:e-4,qan:e-2,mlp:e-5,relu_eff:e-2
        wd = 0.01

        # t, data = generate_periodic_data(int(1667*PERIOD), PERIOD)
        # t_test, data_test = generate_periodic_data(667, PERIOD, is_train = False)
        t = np.linspace(-6, 6, 600)
        data = np.array([target_function(x_) for x_ in t])
        t_test = np.linspace(-6*2, 6*2, 600)
        data_test = np.array([target_function(x_) for x_ in t_test])
        y_uper = 1.5
        y_lower = -1.5
    
    # ----------------------------------------------------------------------------------------------------------
    elif periodic_type == 'd5_s1':
        def generate_periodic_data(num_samples, num_periods=100, is_train = True):
            if is_train:
                t = np.linspace(-num_periods * np.pi, num_periods * np.pi, num_samples)
            else:
                t = np.linspace(-num_periods * 3 * np.pi, num_periods * 3 * np.pi, num_samples)
            data = np.sin(t) 
            return t, data
        degree = 5 # degree of the target function
        scaling = 1 # scaling of the data
        coeffs = [0.15 + 0.15j] * degree  # coefficients of non-zero frequencies
        coeff0 = 0.1  # coefficient of zero frequency


        def target_function(x):
            """Generate a truncated Fourier series, where the data gets re-scaled."""
            
            res = coeff0
            for idx, coeff in enumerate(coeffs):
                exponent = np.complex128(scaling * (idx + 1) * x * 1j)
                conj_coeff = np.conjugate(coeff)
                res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
            return np.real(res)
        
        print(f'generate data from the {periodic_type} function')

        PERIOD = 1
        BATCHSIZE = 16
        NUMEPOCH = 501
        PRINTEPOCH = 100
        lr = 1e-1 # FAN调小:e-5,siren:e-,qan:e-1,relu_rff:e-3
        wd = 0.01

        # t, data = generate_periodic_data(int(1667*PERIOD), PERIOD)
        # t_test, data_test = generate_periodic_data(667, PERIOD, is_train = False)
        t = np.linspace(-6, 6, 600)
        data = np.array([target_function(x_) for x_ in t])
        t_test = np.linspace(-6*2, 6*2, 1200)
        data_test = np.array([target_function(x_) for x_ in t_test])
        y_uper = 3
        y_lower = -3
    
    elif periodic_type == 'sin': # d1_s1
        def generate_periodic_data(num_samples, num_periods=100, is_train = True):
            if is_train:
                t = np.linspace(-num_periods * np.pi, num_periods * np.pi, num_samples)
            else:
                t = np.linspace(-num_periods * 3 * np.pi, num_periods * 3 * np.pi, num_samples)
            data = np.sin(t) 
            return t, data
        # 参数设置
        degree = 1  # 仅保留一个频率
        scaling = 1  # 基本频率
        coeffs = [0 + 1j] * degree  # 仅虚部用于生成 sin(x)
        coeff0 = 0  # 无直流偏移

        def target_function(x):
            """Generate a truncated Fourier series, where the data gets re-scaled."""
            
            res = coeff0
            for idx, coeff in enumerate(coeffs):
                exponent = np.complex128(scaling * (idx + 1) * x * 1j)
                conj_coeff = np.conjugate(coeff)
                res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
            return np.real(res)
        
        print(f'generate data from the {periodic_type} function')

        PERIOD = 1
        BATCHSIZE = 16
        NUMEPOCH = 501
        PRINTEPOCH = 100
        lr = 1e-1 # FAN调小:e-5,siren:e-2,qan:e-1,
        wd = 0.01

        # t, data = generate_periodic_data(int(1667*PERIOD), PERIOD)
        # t_test, data_test = generate_periodic_data(667, PERIOD, is_train = False)
        t = np.linspace(-6, 6, 600)
        data = np.array([target_function(x_) for x_ in t])
        t_test = np.linspace(-6*2, 6*2, 1200)
        data_test = np.array([target_function(x_) for x_ in t_test])
        y_uper = 3
        y_lower = -3
    
    
    return t, data, t_test, data_test, PERIOD, BATCHSIZE, NUMEPOCH, PRINTEPOCH, lr, wd, y_uper, y_lower


def plot_periodic_data(t, data, t_test, data_test, result, args, epoch, path, y_uper, y_lower):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_test, data_test, color='blue',linestyle='--')
    # plt.plot(t, data, label='Domain of Training Data', color='green')
    plt.plot(t_test, result, color='red')
    plt.tight_layout()
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.axis('off')
    plt.xlim(min(t_test),max(t_test))
    plt.ylim(y_lower, y_uper)
    # plt.legend()
    plt.savefig(f'{path}/epoch{epoch}.png')

def plot_truth_data(t, data, t_test, data_test, path, y_uper, y_lower):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(35, 5))
    plt.plot(t_test, data_test, label='Domain of Test Data', color='blue')
    plt.plot(t, data, label='Domain of Training Data', color='green')
    # plt.plot(t_test, result, label='Model Predictions', color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(min(t_test),max(t_test))
    plt.ylim(y_lower, y_uper)
    # plt.legend()
    plt.savefig(f'{path}/ground_truth.png')
    
def read_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        train_loss = []
        test_loss = []
        for line in lines:
            if 'Train Loss' in line:
                train_loss.append(float(line.split(' ')[-1].strip()))
            elif 'Test Loss' in line:
                test_loss.append(float(line.split(' ')[-1].strip()))
    return train_loss, test_loss

def plot_periodic_loss(log_file_path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    train_log_loss, test_log_loss = read_log_file(log_file_path)
    
    log_file_name = log_file_path.split('.')[0]
    ax1.plot(np.arange(0,len(train_log_loss)*50,50),train_log_loss, label=log_file_name)
    ax2.plot(np.arange(0,len(test_log_loss)*50,50),test_log_loss, label=log_file_name)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend(loc='upper right')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.legend(loc='upper right')
    plt.savefig(f'{log_file_name}.pdf')
