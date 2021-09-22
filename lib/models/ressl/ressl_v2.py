from .ressl import ReSSL
import torch
from torch import Tensor


class ResslV2(ReSSL):

    def forward(self, im_q: Tensor, im_k: Tensor):
        """
        Input:
            im_q: contrastive augmented image
            im_k: weak augmented image
        Output:
            logitsq, logitsk
        """

        q: Tensor = self.encoder_q(im_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im)  # keys: NxC
            k: Tensor = self._batch_unshuffle_ddp(k, idx_unshuffle)

        logitsq = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logitsk = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        qk = (q * k).sum(dim=1).mean()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logitsq, logitsk, qk
