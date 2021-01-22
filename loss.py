import torch
import torch.nn as nn

global device_

class LocalLossMAE_v3(nn.Module):
    def __init__(self, target_orr, map_fn=lambda x:x, alpha=0., sigma=0.25):
        super(LocalLossMAE_v3, self).__init__()
        self.ref_user = target_orr
        self.map_fn = map_fn
        self.alpha = alpha
        self.sigma = sigma

    # must feed X as is in 'original' space
    def forward(self, x, omega, y_pred, y):
        err = torch.abs(y_pred - y)
        x = self.map_fn(x)
        sim = torch.sum(self.ref_user * x, 1) / ( torch.sqrt(torch.sum((self.ref_user) ** 2)) * torch.sqrt(torch.sum((x) ** 2, 1)) )
        sim = torch.exp(-0.5 * torch.pow((sim - 1) / self.sigma, 2))

        return torch.mean(err * sim) + torch.sum(torch.abs(omega)) * self.alpha


class LocalLossMAE_v2(nn.Module):
    def __init__(self, target_orr, G, alpha=0.0001):
        super(LocalLossMAE_v2, self).__init__()

        self.ref_user = target_orr
        self.alpha = alpha
        self.G = G

    # must feed X as is in 'original' space
    def forward(self, x, sigma, y_pred, y):
        err = torch.abs(y_pred - y)
        x = x @ self.G
        sim = torch.sum(self.ref_user * x, 1) / (
                    torch.sqrt(torch.sum((self.ref_user) ** 2)) * torch.sqrt(torch.sum((x) ** 2, 1)))
        decay = 0.25
        sim = torch.exp(-0.5 * torch.pow((sim - 1) / decay, 2))

        return torch.mean(err * sim) + torch.sum(torch.abs(sigma)) * self.alpha