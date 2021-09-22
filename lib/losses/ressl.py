from torch import nn, Tensor
import torch.nn.functional as F
import torch


class ResslLoss(nn.Module):

    def __init__(self, T_student: float = 0.1, T_teacher: float = 0.04):
        super().__init__()
        self.T_student = T_student
        self.T_teacher = T_teacher

    def forward(self, logits_q: Tensor, logits_k: Tensor) -> Tensor:
        # FIXME: 这里和原版代码不一样，但是和论文一致。我怀疑作者提供的代码是错的。
        loss = - torch.sum(F.softmax(logits_k.detach() / self.T_teacher, dim=1)
                           * F.log_softmax(logits_q / self.T_student, dim=1), dim=1).mean()
        return loss


class ResslLossV2(nn.Module):

    def __init__(self, T_student: float = 0.1, T_teacher: float = 0.04):
        super().__init__()
        self.ressl_loss = ResslLoss(T_student=T_student, T_teacher=T_teacher)

    def forward(self, logits_q: Tensor, logits_k: Tensor, qk: Tensor) -> Tensor:
        loss1 = self.ressl_loss(logits_q, logits_k)
        loss2 = 2.0 - 2.0 * qk
        loss = (loss1 + loss2) / 2.0
        return loss
