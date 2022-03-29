import torch
import torch.nn as nn

from torchattacks.attack import Attack


class RMIFGSM(Attack):

    def __init__(self, model, osnmodel,eps=8/255, steps=5, decay=1.0,alpha=0.5):
        super(RMIFGSM, self).__init__("RMIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = self.eps / self.steps
        self.osnmodel = osnmodel
        self.balance = alpha 

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)


        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()
        
        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            osn_outputs = self.osnmodel(adv_images)
            cost = -self.balance*loss(outputs, labels) - (1-self.balance)*loss(osn_outputs, labels)
            
            grad = torch.autograd.grad(cost, adv_images, 
                                       retain_graph=False, create_graph=False)[0]
            
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
