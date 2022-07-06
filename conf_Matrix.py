import torch
import matplotlib.pyplot as plt
import numpy as np


# Entry [i,j] means i is groundtruth and j is predicted!
def plot_confusion_matrix(cm, classes, fontsize=None):
    plt.imshow(cm, plt.cm.Blues, interpolation="nearest")
    if fontsize is None:
        plt.yticks(np.arange(len(classes)), classes)
        plt.xticks(np.arange(len(classes)), classes, rotation=90)
    else:
        plt.yticks(np.arange(len(classes)), classes, fontsize=fontsize)
        plt.xticks(np.arange(len(classes)), classes, rotation=90, fontsize=fontsize)


class ConfMatrix():
    def __init__(self, num_classes, ignore_index=255, update_speed=None, cuda=False):
        self.num_classes = num_classes
        self.cuda = cuda
        self.ignore_index = ignore_index
        self.conf_matrix = torch.zeros([num_classes, num_classes]).long()
        self.update_speed = update_speed
        if cuda:
            self.conf_matrix = self.conf_matrix.cuda()

    def cuda(self):
        self.conf_matrix = self.conf_matrix.cuda()
        self.cuda = True

    def cpu(self):
        self.conf_matrix = self.conf_matrix.cpu()
        self.cuda = False

    def diagonalize_conf_matrix(self):
        old_conf_matrix = self.normalizeConfusion()
        diagonal_matrix = torch.empty_like(old_conf_matrix)

        tmp_size = old_conf_matrix.size()[0]

        for offset in range(tmp_size):
            for i in range(tmp_size):
                diagonal_matrix[offset, i] = old_conf_matrix[i, (i + offset) % tmp_size]

        return diagonal_matrix

    def normalizeConfusion(self):
        normalized_matrix = self.conf_matrix.clone().float()
        sum_vec = torch.sum(normalized_matrix, 1).unsqueeze(1)
        sum_vec[sum_vec == 0] = 1
        normalized_matrix = normalized_matrix / sum_vec

        return normalized_matrix

    def getAbsolutConfusion(self):
        absolute_matrix = self.normalizeConfusion()
        absolute_matrix = (absolute_matrix + absolute_matrix.t()) / 2

        return absolute_matrix

    # Can be used to reset the diagonal to 1 and the rest to 0.
    def init_perfect(self):
        self.conf_matrix.fill_(0)
        for i in range(self.conf_matrix.size(0)):
            self.conf_matrix[i, i] = 1

    # Reset the Matrix
    def init_zero(self):
        self.conf_matrix.fill_(0)

    def addPred(self, mask, pred):
        self.conf_matrix += self._compute_update(mask, pred)

    def _compute_update(self, old_mask, pred):
        mask = old_mask.clone()
        mask[mask == self.ignore_index] = self.num_classes
        conf = torch.bincount(self.num_classes * mask.view(mask.numel()) + pred.view(pred.numel()),
                              minlength=(self.num_classes + 1) ** 2)

        return conf[:-(2 * self.num_classes + 1)].view(self.num_classes, self.num_classes)

    def getFrequenzies(self):
        return torch.sum(self.conf_matrix, 1).float() / torch.sum(self.conf_matrix).float()

    def getClassIoU(self, class_id=None):
        if class_id is not None:
            return self.conf_matrix[class_id, class_id].float() / (
                        torch.sum(self.conf_matrix[:, class_id]).float() + torch.sum(
                    self.conf_matrix[class_id, :]).float() - self.conf_matrix[class_id, class_id]).float()
        else:
            return self.conf_matrix.diag().float() / (
                        torch.sum(self.conf_matrix, 0).float() + torch.sum(self.conf_matrix,
                                                                           1).float() - self.conf_matrix.diag().float())

    def getClassDice(self, class_id=None):
        if class_id is not None:
            return (2 * self.conf_matrix[class_id, class_id].float()) / (
                        torch.sum(self.conf_matrix[:, class_id]).float() + torch.sum(
                    self.conf_matrix[class_id, :]).float())
        else:
            return 2 * self.conf_matrix.diag().float() / (
                        torch.sum(self.conf_matrix, 0).float() + torch.sum(self.conf_matrix, 1).float() + 1e-15)

    def getClassAccuracy(self, class_id=None):
        if class_id is not None:
            return self.normalizeConfusion()[class_id, class_id]
        else:
            return self.normalizeConfusion().diag()

    def getClassPrecision(self, class_id=None):
        if class_id is not None:
            return self.conf_matrix[class_id, class_id] / torch.sum(self.conf_matrix[:, class_id]).float()
        else:
            return 0

    def getClassRecall(self, class_id=None):
        if class_id is not None:
            return self.conf_matrix[class_id, class_id] / torch.sum(self.conf_matrix[class_id, :]).float()
        else:
            return 0

    def getAccuracy(self):
        return torch.sum(self.conf_matrix.diag()).float() / torch.sum(self.conf_matrix).float()

    def getPrecision(self):
        return torch.sum(self.conf_matrix.diag()).float() / torch.sum(torch.tril(self.conf_matrix, diagonal=0)).float()

    def getRecall(self):
        return torch.sum(self.conf_matrix.diag()).float() / torch.sum(
            torch.tril(self.conf_matrix.T, diagonal=0)).float()

    def getMIoU(self):
        IoUs = self.getClassIoU()

        return torch.mean(IoUs[torch.isnan(IoUs) == 0])

    def getDice(self):
        Dice = self.getClassDice()

        return torch.mean(Dice[torch.isnan(Dice) == 0])

    def getfIoU(self):
        freq = self.getFrequenzies()
        IoUs = self.getClassIoU()
        nan_ind = torch.isnan(IoUs) == 0

        return torch.sum(freq[nan_ind] * IoUs[nan_ind])


if __name__ == "__main__":
    CM = ConfMatrix(10, ignore_index=-1)
    pred = torch.range(0, 20).long()
    target = torch.range(0, 20).long()
    target[target > 10] = -1
    pred[9] = 5
    pred[3] = 5
    pred[2] = 5
    CM.addPred(target, pred)
    print("Accuracy", CM.getAccuracy())
    print("Recall", CM.getRecall())
    print("Precision", CM.getPrecision())
    print(CM.conf_matrix)
