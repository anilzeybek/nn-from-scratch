import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.arch = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        return self.arch(x)


def generate_adder_data(n_samples=32, max_value=50):
    a = np.random.randint(0, max_value, n_samples)
    b = np.random.randint(0, max_value, n_samples)
    sum = a + b

    return np.column_stack((a, b)), sum


def main():
    model = Model(input_size=2, output_size=1)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)

    for i in range(10000):
        inp, label = generate_adder_data(n_samples=32)

        inp = torch.tensor(inp, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        pred = model(inp).squeeze()
        loss = F.mse_loss(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"epoch: {i+1} loss: {loss.item()}")

    print("==========")

    inp, label = generate_adder_data(n_samples=5)
    inp = torch.tensor(inp, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)

    out = model(inp)
    print("PREDICTIONS:")
    for i in range(inp.shape[0]):
        a = int(inp[i][0].item())
        b = int(inp[i][1].item())
        pred = round(out[i].item())

        print(f"{a} + {b} = {pred}, real={a+b}")


if __name__ == "__main__":
    main()
