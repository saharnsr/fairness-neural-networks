import torch
import torch.nn as nn
import torch.optim as optim


class FairMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def demographic_parity_gap(outputs, sensitive_attr):
    g0 = outputs[sensitive_attr == 0]
    g1 = outputs[sensitive_attr == 1]
    return torch.abs(g0.mean() - g1.mean())


def train_one_step():
    model = FairMLP(10, 32, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Toy data (for demonstration only)
    x = torch.randn(128, 10)
    y = torch.randint(0, 2, (128, 1)).float()
    s = torch.randint(0, 2, (128,))

    logits = model(x)
    loss_pred = criterion(logits, y)
    loss_fair = demographic_parity_gap(logits.squeeze(), s)

    loss = loss_pred + 0.1 * loss_fair

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    train_one_step()

