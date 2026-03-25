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
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        model = ToyModel(32,16)
        model.to("cuda")
        print("参数精度：")
        print("fc1",model.fc1.weight.dtype)
        print("ln",model.ln.weight.dtype)
        print("fc2",model.fc2.weight.dtype)
        out = model(torch.rand(32).cuda())
        print("logits:",out.dtype)
        loss = nn.functional.cross_entropy(out,torch.rand(16).cuda())
        print("loss:",loss.dtype)
        loss.backward()

        print("model grad:",model.fc1.weight.grad.dtype)


