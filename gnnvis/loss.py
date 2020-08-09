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


class UMAPNEGLoss(nn.Module):
    def __init__(self, iscuda, n_neg, batch_size):
        super(UMAPNEGLoss, self).__init__()
        self.iscuda = iscuda
        self.n_neg = n_neg
        self.batch_size = batch_size
        # self._a, self._b = self.find_a_b()
        self._a, self._b = 0.25, 1.0

    def find_a_b(self, spread=1.0, min_dist=0.1):
        from scipy.optimize import curve_fit
        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, covar = curve_fit(curve, xv, yv)
        return params[0], params[1]

    def q_dis(self, dis):
        return 1 / (1 + self._a * torch.pow(dis, self._b))

    def CrossEntropy(self, P, Q):
        if self.iscuda:
            CE = (torch.log(torch.div(P + 1e-12, Q.cpu() + 1e-12)) * P) + (
                        torch.log(torch.div(1 - P + 1e-12, 1 - Q.cpu() + 1e-12)) * (1 - P))
        else:
            CE = (torch.log(torch.div(P + 1e-12, Q + 1e-12)) * P) + (
                        torch.log(torch.div(1 - P + 1e-12, 1 - Q + 1e-12)) * (1 - P))
        return CE

    def forward(self, logits, xs, ys, y_neg, sample_edge_weight, n_neg=20, gamma=7.0):
        xv = logits[xs, :]
        yv = logits[ys, :]
        ynv = logits[y_neg, :]
        sample_edge_weight = torch.FloatTensor(sample_edge_weight)
        dis_neg = torch.zeros((self.batch_size, self.n_neg))

        if self.iscuda:
            sample_edge_weight = sample_edge_weight.cuda()
            dis_neg = dis_neg.cuda()

        for i in range(self.n_neg):
            dis_neg[:, i] = torch.sum(torch.pow(xv - ynv[:, i, :], 2), axis=1).view(-1, )

        dis_pos = torch.sum(torch.pow(xv - yv, 2), axis=1).view(-1, )

        p_pos = self.q_dis(dis_pos)
        p_neg = self.q_dis(dis_neg)

        p_pos = torch.clamp(p_pos, 1e-12, 1 - 1e-12)
        p_neg = torch.clamp(p_neg, 1e-12, 0.99)
        p_pos = sample_edge_weight * torch.log(torch.div(sample_edge_weight, p_pos)) * self.n_neg
        p_neg = (sample_edge_weight - 1).expand((self.n_neg, self.batch_size)).T * torch.log(torch.div(p_neg - 1, (sample_edge_weight - 1).expand((self.n_neg, self.batch_size)).T))

        loss = torch.sum(p_neg, dim=1)
        loss += p_pos

        loss = loss.sum()
        return loss


class LargeVisLoss(nn.Module):
    def __init__(self, iscuda, n_neg, batch_size, pos_times=20, neg_times=1):
        super(LargeVisLoss, self).__init__()
        self.iscuda = iscuda
        self.batch_size = batch_size
        self.n_neg = n_neg
        self.pos_times = pos_times
        self.neg_times = neg_times

    def f1(self, x: torch.Tensor, a: float) -> np.ndarray:
        return 1 / (1 + a * x)

    def f2(self, x: torch.Tensor) -> np.ndarray:
        return 1 / (1 + torch.exp(x))

    def forward(self, logits, xs, ys, y_neg, sample_edge_weight, n_neg=20, gamma=7.0):
        xv = logits[xs, :]
        yv = logits[ys, :]
        ynv = logits[y_neg, :]
        sample_edge_weight = torch.FloatTensor(sample_edge_weight)
        dis_neg = torch.zeros((self.batch_size, self.n_neg))

        if self.iscuda:
            sample_edge_weight = sample_edge_weight.cuda()
            dis_neg = dis_neg.cuda()

        for i in range(self.n_neg):
            dis_neg[:, i] = torch.sum(torch.pow(xv - ynv[:, i, :], 2), axis=1).view(-1, )

        dis_pos = torch.sum(torch.pow(xv - yv, 2), axis=1).view(-1, )

        # p_pos = self.f2(dis_pos)
        # p_neg = self.f2(dis_neg)
        p_pos = self.f1(dis_pos, 0.25)
        p_neg = self.f1(dis_neg, 0.25)

        p_pos = torch.clamp(p_pos, 1e-12, 1 - 1e-12)
        p_neg = torch.clamp(p_neg, 1e-12, 0.99)
        p_pos = torch.log(p_pos) * self.pos_times
        p_neg = torch.log(1 - p_neg) * self.neg_times
        # print(torch.min(p_pos))
        # print(torch.min(p_neg))
        loss = torch.sum(gamma * p_neg, dim=1)
        loss += p_pos

        loss = (sample_edge_weight * loss).sum()

        return - loss
