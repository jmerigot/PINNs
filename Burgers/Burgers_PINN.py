from typing import List, Tuple
import torch
from torch import nn, autograd, Tensor
from torch.nn import functional as F

import json
from pathlib import Path
import random
from time import time
import scipy
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm


def calc_grad(y, x) -> Tensor:
    grad = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad

class BurgersPiNN(nn.Module):
    """
    Burger's is in 1D, inputs are x, t.
    As opposed to 3D Navier-Stokes which has inputs x, y, t
    """
    def __init__(self, hidden_dims: List[int], nu: float):
        super(BurgersPiNN, self).__init__()
        self.hidden_dims = hidden_dims
        self.nu = nu # diffusion term
        self.ffn_layers = []
        input_dim = 2 # inputs are x, t
        for hidden_dim in hidden_dims:
            self.ffn_layers.append(nn.Linear(input_dim, hidden_dim))
            self.ffn_layers.append(nn.Tanh())
            self.ffn_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        self.ffn_layers.append(nn.Linear(input_dim, 1)) # output is u
        self.ffn = nn.Sequential(*self.ffn_layers)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: Tensor, t: Tensor, u: Tensor = None):
        inputs = torch.stack([x, t], dim=1)
        u_pred = self.ffn(inputs).squeeze(1)
        
        # compute gradients
        u_t = calc_grad(u_pred, t)
        u_x = calc_grad(u_pred, x)
        u_xx = calc_grad(u_x, x)
        
        # Burger's PDE
        f_u = u_t + u_pred * u_x - self.nu * u_xx
        
        loss = None
        if u is not None:
            loss = self.loss_fn(u, u_pred, f_u)
        
        return {
            "u_pred": u_pred,
            "f_u": f_u,
            "loss": loss
        }
    
    def loss_fn(self, u, u_pred, f_u_pred):
        loss = (
            F.mse_loss(u, u_pred, reduction="sum")
            + F.mse_loss(f_u_pred, torch.zeros_like(f_u_pred), reduction="sum")
        )
        return loss
    
    
    
class BurgersPiNNDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data
        self.examples = torch.tensor(
            data, dtype=torch.float32, requires_grad=True
        )
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headers = ["t", "x", "u"]
        return {key: self.examples[idx, i] for i, key in enumerate(headers)}

def load_jsonl(path, skip_first_lines: int = 0):
    with open(path, "r") as f:
        for _ in range(skip_first_lines):
            next(f)
        return [json.loads(line) for line in f]


def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_dataset(data_path: Path) -> Tuple[BurgersPiNNDataset, BurgersPiNNDataset]:
    data = load_jsonl(data_path, skip_first_lines=1)
    random.shuffle(data)

    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_data = BurgersPiNNDataset(np.array(train_data))
    test_data = BurgersPiNNDataset(np.array(test_data))
    return train_data, test_data

def analytical_solution(x, t, U=1.0, alpha=0.01, k=np.pi):
    return U * np.exp(-alpha * k**2 * t) * np.sin(k * x)

def generate_burgers_data(num_points: int, nu: float=0.01, U=1.0, alpha=0.01, k=np.pi):
    x = np.linspace(-1, 1, num_points)
    t = np.linspace(0, 1, num_points)
    X, T = np.meshgrid(x, t)
    X_flat = X.flatten()
    T_flat = T.flatten()
    
    # compute solution using the analytical function
    U_flat = analytical_solution(X_flat, T_flat, U=U, alpha=alpha, k=k)

    # add noise
    #U_flat += 0.01 * np.random.randn(*U_flat.shape)

    data = np.vstack((T_flat, X_flat, U_flat)).T
    return data

def get_burgers_dataset(num_points: int = 10000) -> Tuple[BurgersPiNNDataset, BurgersPiNNDataset]:
    data  = generate_burgers_data(num_points)
    np.random.shuffle(data)
    
    split_idx = int(len(data) * 0.85)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_data = BurgersPiNNDataset(train_data)
    test_data = BurgersPiNNDataset(test_data)
    return train_data, test_data



class Trainer:
    def __init__(self, model: BurgersPiNN):
        self.model = model
        
        # hyperparameters
        self.lr = 0.001
        self.lr_step = 1
        self.lr_gamma = 0.5
        self.num_epochs = 5
        self.batch_size = 256
        self.log_interval = 5
        self.samples_per_epoch = 10000
        
        # for checkpointing
        self.output_dir = Path(
            "result_Burgers",
            f"pinn-bs{self.batch_size}-lr{self.lr}-lrstep{self.lr_step}"
            f"-lrgamma{self.lr_gamma}-epoch{self.num_epochs}",
        )

        print(f"Output dir: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        args = {}
        for attr in ["lr", "lr_step", "lr_gamma", "num_epochs", "batch_size"]:
            args[attr] = getattr(self, attr)
        dump_json(self.output_dir / "args.json", args)

        # device, optimizer, learning rate, scheduler
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.5
        )
        
        self.loss_history = []
        
    def get_last_ckpt_dir(self) -> Path:
        ckpts = list(self.output_dir.glob("ckpt-*"))
        if len(ckpts) == 0:
            return None
        return sorted(ckpts)[-1]

    def train(self, train_data: BurgersPiNNDataset):
        model = self.model
        device = self.device
        
        sampler = RandomSampler(
            train_data, replacement=True, num_samples=self.samples_per_epoch
        )
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, sampler=sampler
        )
        
        print("====== Training ======")
        print(f"# epochs: {self.num_epochs}")
        print(f"# examples: {len(train_data)}")
        print(f"# samples used per epoch: {self.samples_per_epoch}")
        print(f"batch size: {self.batch_size}")
        print(f"# steps: {len(train_loader)}")
        loss_history = []
        
        model.train()
        model.to(device)
        
        # resume training from a checkpoint
        last_ckpt_dir = self.get_last_ckpt_dir()
        if last_ckpt_dir is not None:
            print(f"Resuming from {last_ckpt_dir}")
            model.load_state_dict(torch.load(last_ckpt_dir / "ckpt.pt"))
            self.optimizer.load_state_dict(
                torch.load(last_ckpt_dir / "optimizer.pt")
            )
            self.lr_scheduler.load_state_dict(
                torch.load(last_ckpt_dir / "lr_scheduler.pt")
            )
            ep = int(last_ckpt_dir.name.split("-")[-1]) + 1
        else:
            ep = 0
        
        train_start_time = time()
        losses_x_neg = []
        losses_x_pos = []
        
        while ep < self.num_epochs:
            print(f"====== Epoch {ep} ======")
            for step, batch in tqdm(enumerate(train_loader)):
                x_batch, t_batch, u_batch = batch["x"].to(device), batch["t"].to(device), batch["u"].to(device)
                
                x_values = x_batch.cpu().detach().numpy()
                neg_count = (x_values < 0).sum()
                pos_count = (x_values >= 0).sum()
                #print(f"Epoch {ep}, Batch x values: {x_values}, Neg count: {neg_count}, Pos count: {pos_count}")
                
                self.optimizer.zero_grad()
                
                # forward
                outputs = model(x_batch, t_batch, u_batch)
                loss = outputs["loss"]
                loss_history.append(loss.item())
                
                # backward
                loss.backward()
                self.optimizer.step()
                
                # Log separate losses for visualization
                if neg_count > 0:
                    losses_x_neg.append(loss.item())
                if pos_count > 0:
                    losses_x_pos.append(loss.item())
                
                if step % self.log_interval == 0:
                    print(
                        {
                            "step": step,
                            "loss": round(loss.item(), 6),
                            "loss_x_neg": losses_x_neg[-1],
                            "loss_y_pos": losses_x_pos[-1],
                            "lr": round(self.optimizer.param_groups[0]["lr"], 4),
                            "time": round(time() - train_start_time, 1),
                        }
                    )
            self.lr_scheduler.step()
            self.checkpoint(ep)
            print(f"====== Epoch {ep} done ======")
            ep += 1
        print("====== Training done ======")
        
        # Plotting the losses
        plt.plot(losses_x_neg, label='Loss for x < 0')
        plt.plot(losses_x_pos, label='Loss for x >= 0')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs for Different Regions')
        plt.legend()
        plt.show()
    
    def checkpoint(self, ep: int):
        """
        Dump checkpoint (model, optimizer, lr_scheduler) to "ckpt-{ep}" in
        the `output_dir`,

        and dump `self.loss_history` to "loss_history.json" in the
        `ckpt_dir`, and clear `self.loss_history`.
        """
        # Evaluate and save
        ckpt_dir = self.output_dir / f"ckpt-{ep}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpointing to {ckpt_dir}")
        torch.save(self.model.state_dict(), ckpt_dir / "ckpt.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(self.lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt")
        dump_json(ckpt_dir / "loss_history.json", self.loss_history)
        self.loss_history = []
    
    def predict(self, test_data: BurgersPiNNDataset) -> dict:
        batch_size = self.batch_size
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False
        )
        
        print("====== Testing ======")
        print(f"# examples: {len(test_data)}")
        print(f"batch size: {batch_size}")
        print(f"# steps: {len(test_loader)}")
        
        self.model.to(self.device)
        self.model.eval()
        
        all_preds = []
        all_losses = []
        
        for step, batch in tqdm(enumerate(test_loader)):
            x_batch, t_batch = batch["x"].to(self.device), batch["t"].to(self.device)
            
            outputs = self.model(x_batch, t_batch)
            f_u = outputs["f_u"]
            all_preds.append(outputs["u_pred"])
            
            # compute residual-based loss
            loss = F.mse_loss(f_u, torch.zeros_like(f_u), reduction="sum")
            all_losses.append(loss.item())
            
        print("====== Testing done ======")
        all_preds = torch.cat(all_preds, 0)
        loss = sum(all_losses) / len(all_losses)
        return {
            "loss": loss,
            "preds": all_preds
        }



torch.random.manual_seed(0)
random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# model
hidden_dims = [20] * 8
model = BurgersPiNN(hidden_dims=hidden_dims, nu=0.01)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")



# data
train_data, test_data = get_burgers_dataset()

# train model
trainer = Trainer(model)
trainer.train(train_data)


# predict
predictions = trainer.predict(test_data)



import matplotlib.pyplot as plt
import numpy as np

# Define the analytical solution
def analytical_solution(x, t, U=1.0, alpha=0.01, k=np.pi):
    return U * np.exp(-alpha * k**2 * t) * np.sin(k * x)

# Extracting predictions
u_pred = predictions['preds'].cpu().detach().numpy()

# Assuming the test data contains the inputs x and t in the same order as the predictions
x_test = test_data.examples[:, 1].cpu().detach().numpy()  # x values
t_test = test_data.examples[:, 0].cpu().detach().numpy()  # t values

# Reshape the data for plotting
x_unique = np.unique(x_test)
t_unique = np.unique(t_test)
X, T = np.meshgrid(x_unique, t_unique)

# Interpolate the predictions to the grid
U_pred_grid = np.zeros(X.shape)
for i in range(len(x_test)):
    xi = np.where(x_unique == x_test[i])[0][0]
    ti = np.where(t_unique == t_test[i])[0][0]
    U_pred_grid[ti, xi] = u_pred[i]

# Compute the analytical solution on the same grid
U_analytical_grid = analytical_solution(X, T)

# Visualization

# Contour plot of the predicted velocity field
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
contour_pred = plt.contourf(X, T, U_pred_grid, levels=50, cmap='jet')
plt.colorbar(contour_pred)
plt.title('Predicted Velocity Field (u)')
plt.xlabel('x')
plt.ylabel('t')

# Contour plot of the analytical velocity field
plt.subplot(1, 2, 2)
contour_analytical = plt.contourf(X, T, U_analytical_grid, levels=50, cmap='jet')
plt.colorbar(contour_analytical)
plt.title('Analytical Velocity Field (u)')
plt.xlabel('x')
plt.ylabel('t')

plt.tight_layout()
plt.show()