import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.quasirandom import SobolEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def u_exact(x):
    return torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])

def f_source(x):
    pi = torch.pi
    return 2 * pi**2 * torch.sin(pi * x[:, 0]) * torch.sin(pi * x[:, 1])

def phi(x):
    return x[:, 0] * (1.0 - x[:, 0]) * x[:, 1] * (1.0 - x[:, 1])

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, depth=4, out_dim=1, act=nn.SiLU):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(act())
        for _ in range(depth-1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(act())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

class RitzModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn_u = MLP(in_dim=2, hidden=64, depth=4, out_dim=1, act=nn.SiLU)
    def forward(self, x):
        u_theta = self.nn_u(x)            # (N,1)
        return u_theta * phi(x).view(-1,1)

def energy_loss(model, x):
    x = x.clone().detach().requires_grad_(True)
    u = model(x)                       
    u_x = torch.autograd.grad(outputs=u, inputs=x, 
                              grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    grad_sq = (u_x**2).sum(dim=1)      
    J1 = 0.5 * torch.mean(grad_sq)     
    J2 = torch.mean(f_source(x) * u.squeeze(1))  
    energy = J1 - J2
    return energy

def sample_sobol(n):
    engine = SobolEngine(dimension=2, scramble=True)
    return engine.draw(n).to(device)

def train(model, epochs=2000, batch_size=4096, lr=1e-3, use_lbfgs=True):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(1, epochs+1):
        x = sample_sobol(batch_size).requires_grad_(True)
        loss = energy_loss(model, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if ep % 200 == 0 or ep == 1:
            print(f"Epoch {ep}/{epochs} | energy = {loss.item():.6e}")
    if use_lbfgs:
        print("Refining with LBFGS...")
        def closure():
            optimizer_lbfgs.zero_grad()
            x = sample_sobol(batch_size).requires_grad_(True)
            loss_lb = energy_loss(model, x)
            loss_lb.backward()
            return loss_lb
        optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.5, max_iter=200)
        optimizer_lbfgs.step(closure)
    return losses

# # usage example:
# model = RitzModel().to(device)
# losses = train(model, epochs=2000, batch_size=4096, lr=1e-3, use_lbfgs=True)
# torch.save(model.state_dict(), f'model_ritz.pth')

model = RitzModel()
model.load_state_dict(torch.load(f'model_ritz.pth', \
                                     weights_only=False))

# evaluate on grid
nx = 101
grid = torch.stack(torch.meshgrid(torch.linspace(0,1,nx), torch.linspace(0,1,nx)), dim=-1).view(-1,2).to(device)
with torch.no_grad():
    u_pred = model(grid).cpu().numpy().reshape(nx, nx)
    u_ex = u_exact(grid).cpu().numpy().reshape(nx, nx)
    err = np.abs(u_pred - u_ex)
    l2 = np.sqrt(np.mean((u_pred - u_ex)**2))
print(f"L2 error on grid: {l2:.6e}")


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

error_2d = np.abs(u_pred - u_ex)

x = torch.linspace(0,1,nx).numpy()
y = torch.linspace(0,1,nx).numpy()
X, Y = np.meshgrid(x, y)
# 预测解可视化
fig = plt.figure(figsize=(18, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, u_pred.T, cmap='viridis')
ax1.set_title("Predicted Solution $u_{pre}$")

# 精确解可视化
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, u_ex.T, cmap='viridis')
ax2.set_title("Exact Solution $u_{exact}$")

# 误差可视化
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, error_2d.T, cmap='hot')
ax3.set_title("Absolute Error $|u_{pre} - u_{exact}|$")

plt.tight_layout()
plt.show()
