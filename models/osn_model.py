import torch.nn as nn
import scseunet
import numpy as np
import torch 


class OSN_Net(nn.Module):

    def __init__(self,tmodel,patch_size=256):
        super(OSN_Net,self).__init__()
        self.name ='OSN_Net'
        self.patch_size = patch_size
  
        self.UNet = scseunet.SCSEUnet(seg_classes=3,backbone_arch='senet154')
        self.UNet =torch.nn.DataParallel(self.UNet).cuda() 
        modelpath='.pth' 
        self.modelname = 'scseunet'
        self.UNet.load_state_dict(torch.load(modelpath))
        print("load {}".format(modelpath))
               
        self.target_model = tmodel
      
    def forward(self, x):

        self.mean = torch.Tensor(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis])
        self.std = torch.Tensor(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis])
        height,width = x.shape[-2],x.shape[-1]
        self.mean = self.mean.expand(3, int(height), int(width)).cuda()
        self.std = self.std.expand(3, int(height), int(width)).cuda()
        x = (x-self.mean)/self.std #x has to normalized to [-1,1]
        x1 = self.UNet(x)
        x2 = x1 * self.std + self.mean
        x3 = torch.clamp(x2,0.,1.)
        outputs = self.target_model(x3)

        return outputs

class OSN_Model(nn.Module):
    def __init__(self,target_model,patch_size=256):
        super(OSN_Model,self).__init__()
        self.model = OSN_Net(target_model,patch_size)
        self.model = torch.nn.DataParallel(self.model).cuda() 

    def forward(self, x):
        self.model.eval()
        outputs = self.model(x)
        return outputs

            
    

