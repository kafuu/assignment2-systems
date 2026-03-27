import torch
import torch.nn as nn


def mix_pricision():
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)

    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

class ToyModel(nn.Module):
        
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("过程精度：")
        x = self.fc1(x)
        x = self.relu(x)
        print("fc1:",x.dtype)
        x = self.ln(x)
        print("ln:",x.dtype)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":

    x = torch.tensor([10, 20, 30])

    print("x:", x)
    print("x.shape:", x.shape)

    print("\nx[:, None]:")
    print(x[:, None])
    print(x[:, None].shape)

    print("\nx[None, :]:")
    print(x[None, :])
    print(x[None, :].shape)

    print("\nunsqueeze(0):")
    print(x.unsqueeze(0))
    print(x.unsqueeze(0).shape)

    print("\nunsqueeze(1):")
    print(x.unsqueeze(1))
    print(x.unsqueeze(1).shape)


