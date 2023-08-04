import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.arch = nn.Sequential(
            nn.Linear(input_size, 2),
            nn.Sigmoid(),
            nn.Linear(2, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.arch(x)


def main():
    data = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=torch.float32,
    )

    network = Model(input_size=2, output_size=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(params=network.parameters(), lr=1e-1)

    for _ in range(100000):
        pred = network(data[:, :-1]).squeeze()
        loss = loss_fn(pred, data[:, -1])

        print(f"loss: {loss.data}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for i in data:
        print(f"{i[0]} ^ {i[1]} = {network(i[:2]).item():.3f}")


if __name__ == "__main__":
    main()
