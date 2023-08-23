import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.arch = nn.Sequential(nn.Linear(input_size, 2), nn.Sigmoid(), nn.Linear(2, output_size), nn.Sigmoid())

    def forward(self, x):
        return self.arch(x)


def main():
    torch.random.manual_seed(0)
    data = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=torch.float32,
    )

    model = NN(input_size=2, output_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    in_data = data[:, :-1]
    label = data[:, -1]

    for i in range(2000):
        pred = model(in_data).squeeze()

        loss = F.mse_loss(pred, label)
        loss.backward()

        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"epoch: {i+1} loss: {loss}")

    for i in data:
        print(f"{i[0]} ^ {i[1]} = {model(i[:2]).item():.3f}")


if __name__ == "__main__":
    main()
