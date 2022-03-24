import torchvision.models as tmodel 
import torch
import os 
from torchvision import transforms


class Resnet50_Imagenet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.name='Resnet50_Imagenet'
        self.model = tmodel.resnet50(pretrained=True)
        self.model = torch.nn.DataParallel(self.model).cuda()
        
        self._mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
        self._std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()

    def forward(self, x):
        self.model.eval()
        #x = x.transpose(1, 2).transpose(1, 3).contiguous()
        input_var = (x.cuda() - self._mean_torch) / self._std_torch
        labels = self.model(input_var)

        return labels


class VGG19bn_Imagenet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.name = 'VGG19bn_Imagenet'
        self.model = tmodel.vgg19_bn(pretrained=True)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self._mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
        self._std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
        self.transform = transforms.Compose([transforms.Resize((224, 224))])
    def forward(self, x):
        self.model.eval()
        #x = x.transpose(1, 2).transpose(1, 3).contiguous()
        #x = self.transform(x)
        input_var = (x.cuda() - self._mean_torch) / self._std_torch
        labels = self.model(input_var)

        return labels


class IncepV3_Imagenet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.name='IncepV3_Imagenet'
        self.model = tmodel.inception_v3(pretrained=True)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self._mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
        self._std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
        self.transform = transforms.Compose([transforms.Resize((299, 299))])
    def forward(self, x):
        self.model.eval()
        #x = x.transpose(1, 2).transpose(1, 3).contiguous()
        #x = self.transform(x)
        input_var = (x.cuda() - self._mean_torch) / self._std_torch
        labels = self.model(input_var)
        return labels