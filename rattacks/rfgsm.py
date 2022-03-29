import torch
import torch.nn as nn
from torchattacks.attack import Attack


class RFGSM(Attack):

    def __init__(self, model,osnmodel, eps=0.007,alpha=0.5):
        super(RFGSM, self).__init__("RFGSM", model)
        self.eps = eps
        self.osnmodel = osnmodel
        self.alpha = alpha

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)
        outputs_osn = self.osnmodel(images)
        cost = -self.alpha*loss(outputs, labels) - (1-self.alpha)*loss(outputs_osn, labels)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images - self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
