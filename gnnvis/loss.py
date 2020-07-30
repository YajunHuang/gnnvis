import torch
import torch.nn as nn


class TSNELoss(nn.Module):
    def __init__(self, iscuda):
        super(TSNELoss, self).__init__()
        self.iscuda = iscuda

    def t_student_dis(self, scores, n):
        I = torch.eye(n).cuda() if self.iscuda else torch.eye(n)
        sum_X = torch.sum(torch.pow(scores, 2), 1)
        D2 = torch.add(torch.add(-2 * torch.mm(scores, scores.t()), sum_X).t(),
                       sum_X)  # np.dot(X,X.T)
        t_dist = 1 / (1 + D2)
        sum_t_dist = (torch.sum(t_dist) - n)
        Q = torch.div(t_dist - I, sum_t_dist) + I
        Q = torch.max(Q, torch.zeros_like(Q).fill_(1e-12))
        return Q

    def KLD(self, P, Q, n):
        I = torch.eye(n).cuda() if self.iscuda else torch.eye(n)
        KLD = ((torch.log(P + I) - torch.log(Q)) * P)
        return KLD

    def forward(self, logits, P):
        n = len(logits)
        Q = self.t_student_dis(logits, n)
        kld = self.KLD(P, Q, n)
        loss_train = kld.sum()
        return loss_train


class UMAPLoss(nn.Module):
    def __init__(self, iscuda):
        super(UMAPLoss, self).__init__()
        self.iscuda = iscuda

    def t_student_dis(self, scores, n):
        I = torch.eye(n).cuda() if self.iscuda else torch.eye(n)
        sum_X = torch.sum(torch.pow(scores, 2), 1)
        D2 = torch.add(torch.add(-2 * torch.mm(scores, scores.t()), sum_X).t(),
                       sum_X)  # np.dot(X,X.T)
        t_dist = 1 / (1 + D2)
        sum_t_dist = (torch.sum(t_dist) - n)
        Q = torch.div(t_dist - I, sum_t_dist) + I
        return Q

    def CrossEntropy(self, P, Q, n):
        I = torch.eye(n).cuda() if self.iscuda else torch.eye(n)
        CE = ((torch.log(P + I) - torch.log(Q)) * P) + (
                    (torch.log(1 - P) - torch.log(1 - (Q - I))) * (1 - P))
        return CE

    def forward(self, logits, P, mamap, mimap, neg_sample_ratio=1.0):
        n = len(logits)
        Q = self.t_student_dis(logits, n)
        loss_train = self.CrossEntropy(P, Q, n)
        loss_train = neg_sample_ratio * loss_train.mul(mimap) + loss_train + (loss_train.mul(mamap) / neg_sample_ratio)
        return loss_train.sum()


class LargeVisLoss(nn.Module):
    pass