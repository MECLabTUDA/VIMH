import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import make_grid, save_image


def plot_prediction(pred, uncert, img, gt, writer, step, config):
    pred = pred.detach().cpu()
    uncert = uncert.detach().cpu()
    img = img.detach().cpu()
    gt = gt.detach().cpu()

    pred = pred.float() / 3.0
    gt = gt.float() / 3.0
    img = img[:, :3, :, :]
    for i in range(img.size()[0]):
        img[i, :, :, :] -= img[i, :, :, :].min()
        img[i, :, :, :] /= img[i, :, :, :].max()

    uncerts = make_grid(torch.stack(3 * [uncert], 1), nrow=1)
    preds = make_grid(torch.stack(3 * [pred], 1), nrow=1)
    imgs = make_grid(img, nrow=1)
    gts = make_grid(torch.stack(3 * [gt], 1), nrow=1)

    all = torch.cat([imgs.float(), gts.float(), preds.float(), uncerts.float()], 2).float()
    writer.add_image('Image', all, step)
    save_image(all, 'runs/%s/val.png' % config.str_name)


def plot_histogram(net, writer, step):
    vars = []
    means = []
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            var = torch.flatten(1e-10 + F.softplus(module.W_p, beta=1, threshold=20))
            vars.append(var.detach().cpu())
            mean = torch.flatten(module.W_mu)
            means.append(mean.detach().cpu())

    vars = torch.cat(vars)
    means = torch.cat(means)

    writer.add_histogram('var', vars.numpy(), step)
    writer.add_histogram('mean', means.numpy(), step)
