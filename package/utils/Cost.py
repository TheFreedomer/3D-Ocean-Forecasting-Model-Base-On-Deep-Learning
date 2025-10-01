import torch.nn as nn
import torch


class MSE(nn.Module):
    """

    """
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        loss = torch.mean((x - y) ** 2)
        return loss


class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true_mean = y_true.mean(dim=0, keepdim=True)
        y_pred_mean = y_pred.mean(dim=0, keepdim=True)

        covariance = ((y_true - y_true_mean) * (y_pred - y_pred_mean)).mean(dim=0)

        std_true = y_true.std(dim=0, unbiased=False)
        std_pred = y_pred.std(dim=0, unbiased=False)

        pearson = covariance / (std_true * std_pred + 1e-8)

        return -pearson.mean()


class DynamicMixedLoss(nn.Module):
    def __init__(self, initial_alpha=0.7):
        super().__init__()
        self.alpha = initial_alpha
        self.pearson_loss = PearsonLoss()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true):
        pearson = self.pearson_loss(y_pred, y_true)
        mse = self.mse_loss(y_pred, y_true).mean(dim=0).mean()

        return self.alpha * mse + (1 - self.alpha) * pearson


if __name__ == '__main__':
    loss_fn = DynamicMixedLoss()
    x_ = torch.randn((5, 1, 3, 224, 224))
    y_ = torch.randn((5, 1, 3, 224, 224))
    loss = loss_fn(x_, y_)
    pass
