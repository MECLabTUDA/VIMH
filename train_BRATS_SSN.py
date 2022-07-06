import torch
import tqdm
from torch import nn
import math
from tensorboardX import SummaryWriter
import os
from importlib.machinery import SourceFileLoader
import argparse

try:
    from .models.unet import UNet, UNet2, StochasticUNet2
    from .conf_Matrix import ConfMatrix
    from .brats import BRATS
    from .utils import plot_prediction, plot_histogram
except:
    from models.unet import UNet, UNet2, StochasticUNet2
    from conf_Matrix import ConfMatrix
    from brats import BRATS
    from utils import plot_prediction, plot_histogram

def train_epoch(net, train_loader, config):
    correct_train = 0

    mean_train_loss = 0.
    mean_train_kl = 0.
    mean_train_entropy = 0.
    conf_mat_train = ConfMatrix(config.num_classes, config.ignore_index, cuda=config.cm_cuda)
    num_train_pixels = 0

    net.train()
    for batch in tqdm.tqdm(train_loader):
        opti.zero_grad()
        img, mask = batch

        img = img.cuda()
        mask = mask.cuda()
        soft_out, _ = net.sample_forward(img, mask, config.num_samples, config.num_classes)
        preds = torch.argmax(soft_out, 1)

        l = loss(torch.log(soft_out + 1e-18), mask)
        l_final = l
        l_final.backward()
        opti.step()
        soft_out = soft_out.detach().cpu()

        pm = preds == mask
        un255 = mask != config.ignore_index
        correct_train += torch.sum(pm & un255).item()
        num_train_pixels += (mask.size(0) * mask.size(1) * mask.size(2)) - torch.sum(mask == 255).item()

        # correct_train += torch.sum(preds == mask)
        mean_train_loss += l
        mean_train_kl += torch.zeros(1)  # kl

        # Compute the entropy of the prediction!
        mean_train_entropy -= torch.sum(torch.sum(soft_out * torch.log(soft_out + 1e-18), 1)[un255])

        conf_mat_train.addPred(mask, preds)

    return correct_train, num_train_pixels, mean_train_loss.cpu().item(), mean_train_kl.cpu().item(), mean_train_entropy.cpu().item(), conf_mat_train.getMIoU().cpu().item()


def eval_epoch(net, val_loader, writer, step, config):
    # Counter
    correct_eval = 0

    mean_eval_loss = 0.
    mean_eval_kl = 0.

    mean_eval_entropy = 0.

    conf_mat_eval = ConfMatrix(config.num_classes, config.ignore_index, config.cm_cuda)

    num_eval_pixels = 0

    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            img, mask = batch

            img = img.cuda()
            mask = mask.cuda()

            if i != len(val_loader) - 2:
                soft_out = net.sample_forward(img, config.num_samples, config.num_classes)
            else:
                soft_out = net.sample_forward(img, config.num_plot_samples, config.num_classes)

            preds = torch.argmax(soft_out, 1)
            l = loss(torch.log(soft_out + 1e-18), mask)

            soft_out = soft_out.detach().cpu()

            pm = preds == mask
            un255 = mask != config.ignore_index
            correct_eval += torch.sum(pm & un255).item()
            num_eval_pixels += (mask.size(0) * mask.size(1) * mask.size(2)) - torch.sum(mask == config.ignore_index).item()

            mean_eval_loss += l

            # Compute the entropy of the prediction!
            mean_eval_entropy -= torch.sum(torch.sum(soft_out * torch.log(soft_out + 1e-18), 1)[un255])

            conf_mat_eval.addPred(mask, preds)

            if i == len(val_loader) - 2:
                uncert = -torch.sum(soft_out * torch.log(soft_out), 1) / torch.log(torch.Tensor([config.num_classes]))
                plot_prediction(preds, uncert, img, mask, val_loader.dataset, writer, step, config)

    return correct_eval, num_eval_pixels, mean_eval_loss.cpu().item(), mean_eval_kl.cpu().item(), mean_eval_entropy.cpu().item(), conf_mat_eval.getMIoU().cpu().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config for training")
    parser.add_argument("CONFIG_PATH", nargs="*", type=str, help="Path to experiment config file",
                        default="./config/SSNs.py")
    args = parser.parse_args()
    config_file = args.CONFIG_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')
    config = SourceFileLoader(config_module, config_file).load_module()

    if not os.path.exists("./saves"):
        os.mkdir("./saves")
    if not os.path.exists("./runs"):
        os.mkdir("./runs")
    if os.path.exists("./saves/" + config.str_name):
        print("Save folder already exists")
    else:
        os.mkdir("./saves/" + config.str_name)

    if os.path.exists("./runs/" + config.str_name):
        print("Run folder already exists")
    else:
        os.mkdir("./runs/" + config.str_name)

    writer = SummaryWriter("./runs/" + config.str_name)

    device = torch.device('cuda')

    train_dataset = BRATS(config.dataset_path, mode='train', subset=config.subset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.num_workers, drop_last=True)

    val_dataset = BRATS(config.dataset_path, mode='val', subset=config.subset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
                                             num_workers=config.num_workers,
                                             drop_last=False)

    net = StochasticUNet2(config.num_classes, config.num_channels).cuda()
    # net = UNet(config.num_classes, config.num_channels).cuda() # for pre-training use UNet

    # SSNs have to trained on a pre-trained network
    # Please pre-train with UNet and then use Unet2 for the modified forward to build SSNs on top
    if config.resume_path is not None:
        dic = torch.load(config.resume_path)
        net.load_state_dict(dic)

    opti = torch.optim.Adam(net.parameters(), config.lr)
    loss = nn.NLLLoss(ignore_index=config.ignore_index).cuda()

    for epoch in range(config.start_epoch, config.epochs):
        if epoch == 0:
            torch.save(net.state_dict(), open("./saves/" + config.str_name + "/" + str(epoch) + ".save", "wb"))
        correct_train, num_train_pixels, mean_train_loss, mean_train_kl, mean_train_entropy, mIoU_train = \
            train_epoch(net, train_loader, config)
        correct_eval, num_eval_pixels, mean_eval_loss, mean_eval_kl, mean_eval_entropy, mIoU_eval = \
            eval_epoch(net, val_loader, writer, epoch, config)

        correct_train /= num_train_pixels
        correct_eval /= num_eval_pixels

        mean_train_loss /= len(train_dataset)
        mean_train_kl /= len(train_dataset)
        mean_eval_loss /= len(val_dataset)
        mean_eval_kl /= len(train_dataset)

        mean_train_entropy /= num_train_pixels
        mean_eval_entropy /= num_eval_pixels

        # Scale the entropy to [0,1)
        mean_train_entropy /= math.log(config.num_classes)
        mean_eval_entropy /= math.log(config.num_classes)

        print("Epoch {}".format(epoch))
        print("\t Accuracy: ")
        print("\t Train: ", correct_train, "    Eval: ", correct_eval)
        print()
        print("\t CE-Loss: ")
        print("\t Train: ", mean_train_loss, "    Eval: ", mean_eval_loss)
        print()
        print("\t KL: ")
        print("\t Train: ", mean_train_kl, "    Eval: ", mean_eval_kl)
        print()
        print("\t Entropy: ")
        print("\t Train: ", mean_train_entropy, "    Eval: ", mean_eval_entropy)
        print()
        print("\t mIoU: ")
        print("\t Train: ", mIoU_train, "    Eval: ", mIoU_eval)
        print()
        print()
        print()

        writer.add_scalar("acc/train", correct_train, epoch)
        writer.add_scalar("acc/eval", correct_eval, epoch)

        writer.add_scalar("ce/train", mean_train_loss, epoch)
        writer.add_scalar("ce/eval", mean_eval_loss, epoch)

        writer.add_scalar("kl/train", mean_train_kl, epoch)
        writer.add_scalar("kl/eval", mean_eval_kl, epoch)

        writer.add_scalar("entropy/train", mean_train_entropy, epoch)
        writer.add_scalar("entropy/eval", mean_eval_entropy, epoch)

        writer.add_scalar("mIoU/train", mIoU_train, epoch)
        writer.add_scalar("mIoU/eval", mIoU_eval, epoch)

        writer.flush()

        if mIoU_train >= config.best_train:
            config.best_train = mIoU_train
            torch.save(net.state_dict(), open("./saves/" + config.str_name + "/" + "best_train" + ".save", "wb"))
            torch.save(opti.state_dict(), open("./saves/" + config.str_name + "/" + "best_train" + ".opti", "wb"))

        if mIoU_eval >= config.best_eval:
            config.best_eval = mIoU_eval
            torch.save(net.state_dict(), open("./saves/" + config.str_name + "/" + "best_eval" + ".save", "wb"))
            torch.save(opti.state_dict(), open("./saves/" + config.str_name + "/" + "best_eval" + ".opti", "wb"))

        if epoch % 1 == 0:
            torch.save(net.state_dict(), open("./saves/" + config.str_name + "/" + str(epoch + 1) + ".save", "wb"))
            torch.save(opti.state_dict(), open("./saves/" + config.str_name + "/" + str(epoch + 1) + ".opti", "wb"))
