# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.functional import Tensor
import torch.nn as nn
from .backbone import BackBone
import copy


class ReSSL(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, backbone: nn.Module, dim=512, K=65536*2, m=0.999, hidden_dim: int = 4096, dim_in: int = 512):
        """
        dim: feature dimension (default: 512)
        K: queue size; number of negative keys (default: 65536*2)
        m: moco momentum of updating key encoder (default: 0.999)
        """
        super().__init__()

        self.K = K
        self.m = m

        # create the encoders
        self.encoder_q = BackBone(
            backbone=backbone,
            hidden_dim=hidden_dim,
            dim=dim,
            dim_in=dim_in,
        )
        self.encoder_k = BackBone(
            backbone=copy.deepcopy(backbone),
            hidden_dim=hidden_dim,
            dim=dim,
            dim_in=dim_in,
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: Tensor):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all, device=x.device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: contrastive augmented image
            im_k: weak augmented image
        Output:
            logitsq, logitsk
        """

        q = self.encoder_q(im_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im)  # keys: NxC
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        logitsq = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logitsk = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logitsq, logitsk


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output
