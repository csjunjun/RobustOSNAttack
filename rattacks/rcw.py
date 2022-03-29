import torch
import torch.nn as nn
import torch.optim as optim

from torchattacks.attack import Attack


class RCW(Attack):

    def __init__(self, model,osnmodel, c=1e-4, kappa=0, steps=1000, lr=0.01,balance=0.5,atttype=0):
        super(RCW, self).__init__("RCW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.osnmodel = osnmodel
        self.balance = balance
        self.atttype = atttype

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)


        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.model(adv_images)
            osn_outputs = self.osnmodel(adv_images)
            #untargeted
            f_loss = self.f(outputs, labels).sum()
            osn_f_loss = self.f(osn_outputs,labels).sum()

            cost = self.c*L2_loss +(self.balance*f_loss+(1-self.balance)*osn_f_loss)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            if self.atttype==0:
                _, pre = torch.max(outputs.detach(), 1)
            elif self.atttype ==1:
                _, pre = torch.max(osn_outputs.detach(), 1)
            correct = (pre == labels).float()


            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        if self._targeted:
            return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)