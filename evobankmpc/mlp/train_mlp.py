
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


def generate_training_data(n_steps=150000, Ts=0.02, save_path='evobankmpc/data/training_data_large.npz'):

    try:
        from evobankmpc.params import ORCA
        from evobankmpc.models import Dynamic
        from evobankmpc.tracks import ETHZ
    except ImportError:
        X = np.random.randn(n_steps, 11).astype(np.float32)
        y = np.random.randn(n_steps, 6).astype(np.float32) * 0.01
        np.savez(save_path, X=X, y=y)
        return

    params = ORCA(control='pwm')
    model = Dynamic(**params)
    track = ETHZ(reference='optimal', longer=True)

    x_init = np.zeros(model.n_states)
    x_init[0], x_init[1] = track.x_init, track.y_init
    x_init[2] = track.psi_init
    x_init[3] = track.vx_init

    states = np.zeros((model.n_states, n_steps+1))
    inputs = np.zeros((model.n_inputs, n_steps))
    states[:, 0] = x_init

    for i in range(n_steps):
        r = np.random.rand()
        t = i * Ts
        if r < 0.5:      
            pwm = np.random.uniform(-0.1, 1.0)
            steer = np.random.uniform(-0.35, 0.35)
        elif r < 0.7:
            pwm = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
            steer = 0.3 * np.sin(2 * np.pi * 0.8 * t)
        elif r < 0.85: 
            pwm = 1.0 if np.random.rand() > 0.5 else -0.1
            steer = 0.3 if np.random.rand() > 0.5 else -0.3
        else:            
            if i > 0:
                params['Df'] *= 0.9999
                params['Dr'] *= 0.9999
                model.Df, model.Dr = params['Df'], params['Dr']
            pwm = 0.5 + 0.2 * np.sin(2 * np.pi * 0.3 * t)
            steer = 0.2 * np.sin(2 * np.pi * 0.5 * t)
        inputs[0, i] = pwm
        inputs[1, i] = steer

        x_next, _ = model.sim_continuous(states[:, i], inputs[:, i:i+1], [0, Ts])
        states[:, i+1] = x_next[:, -1]

    delta_states = states[:, 1:] - states[:, :-1]  # (6, n_steps)

    psi = states[2, :-1]
    vx = states[3, :-1]
    vy = states[4, :-1]
    omega = states[5, :-1]
    steer = inputs[1, :]
    pwm = inputs[0, :]

    ay = vy + vx * omega   

    lf = params['lf']; lr = params['lr']
    alpha_f = steer - np.arctan2((lf * omega + vy), np.abs(vx) + 1e-6)
    alpha_r = np.arctan2((lr * omega - vy), np.abs(vx) + 1e-6)


    X = np.concatenate([
        states[:, :-1].T,           
        inputs.T,                   
        ay.reshape(-1,1),           
        alpha_f.reshape(-1,1),     
        alpha_r.reshape(-1,1),      
    ], axis=1).astype(np.float32)   

    y = delta_states.T.astype(np.float32)


    mask = np.all(np.abs(y) < 10, axis=1)
    X = X[mask]
    y = y[mask]


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, X=X, y=y)



class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = out + residual
        return self.relu(out)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=11, hidden_dims=[128, 128], output_dim=6, dropout=0.2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for dim in hidden_dims:
            layers.append(ResidualBlock(dim, dropout))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def export_weights_for_jax(model, scaler, output_dir='evobankmpc/mlp/'):
    import pickle
    os.makedirs(output_dir, exist_ok=True)

    weights_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.detach().cpu().numpy().T   # (in, out)
            b = module.bias.detach().cpu().numpy()       # (out,)
            weights_list.append(w)
            weights_list.append(b)

    with open(os.path.join(output_dir, 'residual_mlp_weights.pkl'), 'wb') as f:
        pickle.dump(weights_list, f)

    config = {
        'input_dim': model.net[0].in_features,
        'hidden_dims': [128, 128],
        'output_dim': model.net[-1].out_features,
    }
    np.save(os.path.join(output_dir, 'residual_config.npy'), config)

    np.save(os.path.join(output_dir, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(output_dir, 'scaler_scale.npy'), scaler.scale_)
    print(f"{output_dir}residual_mlp_weights.pkl")
    print(f"{output_dir}residual_config.npy")
    print(f"{output_dir}scaler_mean.npy {output_dir}scaler_scale.npy")

def train():
    data_path = 'evobankmpc/data/training_data_large.npz'
    use_existing = False

    if not use_existing or not os.path.exists(data_path):
        print("new...")
        generate_training_data(n_steps=150000, save_path=data_path)

    print("load...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    model = ResidualMLP(input_dim=11, hidden_dims=[128, 128], output_dim=6, dropout=0.2)
    print(f"model parameters: {sum(p.numel() for p in model.parameters())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"{device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    num_epochs = 500
    train_losses = []
    val_losses = []

    print("training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)
        train_loss = epoch_loss / len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), 'evobankmpc/mlp/best_mlp_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    export_weights_for_jax(model, scaler, output_dir='evobankmpc/mlp/')

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MLP Training Loss (Residual)')
    plt.grid(True)
    plt.savefig('evobankmpc/mlp/mlp_loss_residual.png')
    plt.close()
    print(f"training completed, best validation loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    train()