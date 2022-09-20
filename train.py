import os
gpu_ids = '0,2,5'
# gpu_ids = '7'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
import cv2
import copy
import time
import shutil
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau,ExponentialLR
from torchvision import transforms
from models.scseunet import SCSEUnet
# import sys
# sys.path.append("/home/junliu/code/fbattack") 
from models import ImagenetModels
from sklearn.metrics import roc_auc_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from collections import OrderedDict
from matplotlib import pyplot as plt
#from torchsummary import summary
from PIL import Image

SEED = 666666
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

patch_size = '256'
mean = torch.Tensor(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis])
mean = mean.expand(3, int(patch_size), int(patch_size)).cuda()
std = torch.Tensor(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis])
std = std.expand(3, int(patch_size), int(patch_size)).cuda()

base_path = '/home/junliu/code/RobustOSNAttack/'

class GIID_Dataset(Dataset):
    def __init__(self, num=0, file='', choice='train', test_path='', size=int(patch_size)):
        self.num = num
        self.choice = choice
        if self.choice == 'test':
            self.test_path = test_path
            self.filelist = sorted(os.listdir(self.test_path))
        else:
            self.filelist = file

        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
    ])
        self.test_transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
       ])
        self.blur_kernel_list = [3, 5, 7, 9]
        self.size = size

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        if self.choice == 'test':
            return len(self.filelist)
        return self.num

    def load_item(self, idx):
        fname1, fname2,qf = self.filelist[idx]
        isjpeg = 0
        if fname1!='' and  fname1.split('.')[1] == 'jpg':
            isjpeg = 1 
        else:
            jsjpeg = 0
        img = cv2.imread(fname1)[..., ::-1]
        mask = cv2.imread(fname2)[..., ::-1]
        H, W, _ = img.shape
        Hm, Wm, _ = mask.shape
        if Hm != H or Wm != W:
            img = cv2.resize(img, (Wm, Hm))

        H, W, _ = img.shape
        if H < self.size or W < self.size:
            m = self.size / min(H, W)
            img = cv2.resize(img, (int(H * m) + 1, int(W * m) + 1))
            mask = cv2.resize(mask, (int(H * m) + 1, int(W * m) + 1))

        H, W, _ = img.shape
        if self.choice == 'train' and (H != self.size or W != self.size):
            x = 0 if H == self.size else np.random.randint(0, H-self.size)
            y = 0 if W == self.size else np.random.randint(0, W-self.size)

            img = img[x:x + self.size, y:y + self.size, :]
            mask = mask[x:x + self.size, y:y + self.size, :]
        elif self.choice == 'val' and (H != self.size or W != self.size):
            img = img[(H-self.size)//2:(H-self.size)//2+self.size, (W-self.size)//2:(W-self.size)//2+self.size, :]
            mask = mask[(H-self.size)//2:(H-self.size)//2+self.size, (W-self.size)//2:(W-self.size)//2+self.size, :]

        if self.choice == 'train':
            img, mask = self.aug(img, mask)


        img = img.astype('float') / 255.
        mask = mask.astype('float') / 255.
        return self.transform(img), self.transform(img), self.transform(mask),torch.Tensor([float(qf)]).long(),torch.Tensor([int(isjpeg)]).long(),fname1.split('/')[-1]

    def aug(self, img, mask):

        if random.random() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            tmp = random.random()
            if tmp < 0.33:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
            elif tmp < 0.66:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img = cv2.rotate(img, cv2.cv2.ROTATE_180)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_180)

        return img, mask

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class GIID_Model(nn.Module):
    def loadmodel(self,model_path):
        pretrained_net_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in pretrained_net_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.networks.load_state_dict(new_state_dict)

    def __init__(self):
        super(GIID_Model, self).__init__()
        self.lr = 1e-4
        self.networks = SCSEUnet(seg_classes=3,backbone_arch='senet154',isResidual=True,isJPEG=True)  #jun :best performace  6.67 in log_4.txt
        #print("SCSEUnet summary")
        #summary(self.networks.cuda(),input_size=(3,256,256))
        
        pytorch_total_params = sum(p.numel() for p in self.networks.parameters() if p.requires_grad)
        print('Total Params of %s: %d' % (self.networks.name, pytorch_total_params))
        print('learning rate:{}'.format(self.lr))
        print("gen loss: l2_loss")
        with open(base_path+'log_' + gpu_ids[0] + '.txt', 'a+') as f:
            f.write('\n\nTotal Params of %s: %d' % (self.networks.name, pytorch_total_params))
        self.gen = nn.DataParallel(self.networks).cuda()
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.save_dir = base_path+'weights/'
        self.bce_loss = nn.BCELoss()
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
       
        self.modelname='20220324scseunet_facebook'
        self.ce = nn.CrossEntropyLoss()
        self.recgmodel = ImagenetModels.Resnet50_Imagenet() 
        self.a = 0.5

    def process(self, Ii, Mg,qf,isjpeg,truelabel, eva=False):       
        self.gen_optimizer.zero_grad()
        Ii.sub_(mean).div_(std)
        Mg.sub_(mean).div_(std)
        x = [Ii,qf,isjpeg]
        Mo = self(x)
        gen_loss = self.l2_loss(Mo, Mg)
        
        if not eva:
            self.backward(gen_loss)
        return Mo, gen_loss

    def forward(self, Ii):
        return self.gen(Ii)

    def backward(self, gen_loss=None):
        if gen_loss:
            gen_loss.backward(retain_graph=False)
            self.gen_optimizer.step()

    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)    
        torch.save(self.gen.state_dict(), self.save_dir + path + '%s_%s.pth' % (self.networks.name,self.modelname))
    def load(self, path=''):
        
        self.gen.load_state_dict(torch.load(self.save_dir + path + '%s_weights.pth' % self.networks.name))
        


class ForgeryForensics():
    def __init__(self):
        self.train_num = 2700
        self.batch_size = 32        
        self.train_npy = 'your_train_dataset.npy' # format:[uploadfilepath, downloadfilepath,qfofuploadfile]
        self.val_npy = 'your_train_dataset.npy'
        self.train_file = np.load(base_path+'data/' + self.train_npy)
        self.val_file = self.train_file[self.train_num:]
        self.train_file = self.train_file[:self.train_num]

        np.random.shuffle(self.train_file)

        self.train_num = len(self.train_file)
        self.val_num = len(self.val_file)
        train_dataset = GIID_Dataset(self.train_num, self.train_file, choice='train')
        val_dataset = GIID_Dataset(self.val_num, self.val_file, choice='val')

        self.giid_model = GIID_Model().cuda()
        self.n_epochs = 100
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)
         

    def draw(self,listl,figname,scale=True):
        plt.clf()
        plt.xlabel("epoches")
        if len(listl)==1:
            plt.ylabel("val noise MSE")
        else:
            plt.ylabel("L2 loss x 10+e4")
        s =  10000 if scale==True else 1
        for i in range(0,len(listl)):
            x = list(range(0,len(listl[i])))
            if len(listl) == 2 and i==0:
                
                plt.plot(x, [it*s for it in listl[i]],color='blue',label='train')
            elif len(listl) == 2 and i==1:
                plt.plot(x, [it*s for it in listl[i]],color='red',label='val')
            else:
                plt.plot(x, listl[i])
        plt.savefig(base_path+'logs/{}.png'.format(figname))

    def train(self):
        if not os.path.exists(base_path +'logs'):
            os.mkdir(base_path +'logs')
        with open(base_path +'logs/log_' + gpu_ids[0] + '.txt', 'a+') as f:
            f.write('\nTrain/Val with ' + self.train_npy + '/' + self.val_npy + ' with num %d/%d' % (self.train_num, self.val_num))
        cnt, gen_losses, psnr, ssim, mse = 0, [], [], [], []
        psnr_o, ssim_o, mse_o = [], [], []
        best_score = 9999
        scheduler = ReduceLROnPlateau(self.giid_model.gen_optimizer, patience=10, factor=0.5,mode='min')
        gen_losses_mean = []
        val_gen_loss_mean = []
        val_m_jun_mean = []

        for epoch in range(1, self.n_epochs+1):
            val_m_jun = []
            val_gen_losss =[]
            
            for items in self.train_loader:
                cnt += self.batch_size
                self.giid_model.train()
                Ii, Ir, Mg,qf,isjpeg = (item.cuda() for item in items[:-1])  # Ii,Ir is the original image, Mg is the downloaded image
                Ii.sub_(mean).div_(std)
                Mo, gen_loss= self.giid_model.process(Ir, Mg,qf,isjpeg,items[-1])
                gen_losses.append(gen_loss.item())
                Ii, Mg, Mo = self.convert1(Ii), self.convert1(Mg), self.convert1(Mo)
                p, s, m = metric_osn(Mg - Ii, Mo - Ii)

                p_o, s_o, m_o = 0, 0, 0
                psnr.append(p)
                ssim.append(s)
                mse.append(m)
                psnr_o.append(p_o)
                ssim_o.append(s_o)
                mse_o.append(m_o)
                print('epoch:%d Tra (%d/%d): G:%.8f  P: %4.2f -> %4.2f   S: %5.4f -> %5.4f   M: %5.2f -> %5.2f'
                      % (epoch,cnt, self.train_num, np.mean(gen_losses), np.mean(psnr_o)
                      , np.mean(psnr), np.mean(ssim_o), np.mean(ssim), np.mean(mse_o), np.mean(mse)), end='\r')
                
                if cnt % 500 == 0 or cnt >= self.train_num:
                    val_gen_loss, val_p, val_s, val_m, val_p_o, val_s_o, val_m_o = self.val()
                    tmp_score = val_m
                    scheduler.step(tmp_score)

                    # print('Val (%d/%d): G:%5.4f P:%4.2f S:%5.4f M:%6.2f'
                    #       % (cnt, self.train_num, val_gen_loss, val_p, val_s, val_m))
                    print('Val: G:%.8f P: %4.2f -> %4.2f   S: %5.4f -> %5.4f   M: %5.2f -> %5.2f' % (val_gen_loss, val_p_o, val_p, val_s_o, val_s, val_m_o, val_m))
                    self.giid_model.save('latest_' + gpu_ids[0] + '/')

                    if tmp_score < best_score:
                        best_score = tmp_score
                        self.giid_model.save('best_' + gpu_ids[0] + '/')
                    with open(base_path+'logs/log_' + gpu_ids[0] + '.txt', 'a+') as f:
                        f.write('\n(%5d/%5d): Tra: G:%.8f P:%4.2f S:%5.4f M:%6.2f' % (cnt, self.train_num, np.mean(gen_losses), np.mean(psnr), np.mean(ssim), np.mean(mse)))
                        f.write('Val: G:%.8f P: %4.2f -> %4.2f   S: %5.4f -> %5.4f   M: %5.2f -> %5.2f' % (val_gen_loss, val_p_o, val_p, val_s_o, val_s, val_m_o, val_m))
                        # f.write('Val: G:%5.4f P:%4.2f S:%5.4f M:%6.2f ' % (val_gen_loss, val_p, val_s, val_m))
                    val_m_jun.append(val_m)
                    val_gen_losss.append(val_gen_loss)
                    
            val_gen_loss_mean.append(np.mean(val_gen_losss))
            gen_losses_mean.append(np.mean(gen_losses))  
            val_m_jun_mean.append(np.mean(val_m_jun))


            cnt, gen_losses, psnr, ssim, mse = 0, [], [], [], []
            psnr_o, ssim_o, mse_o = [], [], []

            self.draw([gen_losses_mean,val_gen_loss_mean],'train_val_genloss_{}'.format(self.giid_model.modelname))
            self.draw([val_m_jun_mean],'val_mse_{}'.format(self.giid_model.modelname))

           
            
            

    def val(self):
        
        self.giid_model.eval()
        psnr, ssim, mse, gen_losses = [], [], [], []
        psnr_o, ssim_o, mse_o = [], [], []
        

        for cnt, items in enumerate(self.val_loader):
            Ii, Ir, Mg,oriqf,isjpeg = (item.cuda() for item in items[:-1])
            Ii.sub_(mean).div_(std)
            filename = items[-1]
            Mo, gen_loss = self.giid_model.process(Ir, Mg, oriqf,isjpeg,items[-1],eva=True)

            B, C, H, W = Mo.shape
            gen_losses.append(gen_loss.item())
           
            Ii, Mg, Mo = self.convert1(Ii), self.convert1(Mg), self.convert1(Mo) #original , ground truth,predict
            r1, r2, r3 = Mg - Ii, Mo - Ii, Ii - Ii # rel res, generate res,0
            # r1[r1 > 8] = 0
            # r1[r1 < -8] = 0
            # r2[r2 > 4] = 0
            # r2[r2 < 4] = 0
            # r1 = cv2.normalize(r1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
            # r2 = cv2.normalize(r2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
            p, s, m = metric_osn(r1, r2)
            p_o, s_o, m_o = metric_osn(r1, r3)
            # p, s, m = metric_osn(Mg - Ii, Mo - Ii)
            # p_o, s_o, m_o = metric_osn(Mg - Ii, Ii - Ii)
            psnr.append(p)
            ssim.append(s)
            mse.append(m)
            psnr_o.append(p_o)
            ssim_o.append(s_o)
            mse_o.append(m_o)
            
            
            dp = base_path+'output/{}'.format(self.giid_model.modelname) + gpu_ids[0]
            if not os.path.exists(dp):
                os.mkdir(dp)
                print("new folder {}".format(dp))
            with open(dp+'/info.txt','a+') as f:
                f.write('{}:gen_loss:{},p:{},s:{},m:{},p_o:{},s_o:{},m_o:{}\n'.format(filename,gen_loss.item(),p,s,m,p_o,s_o,m_o))
            # print: original file, download file, generate file, download res, generate res
            if cnt < 100:
                for i in range(len(Mo)):
                    rtn = np.ones([H, W*5+40, 3], dtype=np.uint8) * 255
                    rtn[:, :W, :] = Ii[i][..., ::-1]
                    rtn[:, W * 1 + 10:W * 2 + 10, :] = Mg[i][..., ::-1]
                    rtn[:, W * 2 + 20:W * 3 + 20, :] = Mo[i][..., ::-1]
                    resdual_1 = Mg[i][..., ::-1] - Ii[i][..., ::-1] #download real res
                    resdual_2 = Mo[i][..., ::-1] - Ii[i][..., ::-1] #generate

                    resdual_1 = cv2.normalize(resdual_1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
                    resdual_2 = cv2.normalize(resdual_2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
                    resdual = np.concatenate([resdual_1, resdual_2], axis=2)
                    resdual_1 = cv2.cvtColor(resdual[:, :, :3], cv2.COLOR_RGB2GRAY)
                    resdual_2 = cv2.cvtColor(resdual[:, :, 3:], cv2.COLOR_RGB2GRAY)
                    rtn[:, W * 3 + 30:W * 4 + 30, :] = np.concatenate([resdual_1[..., None], resdual_1[..., None], resdual_1[..., None]], axis=2)
                    rtn[:, W * 4 + 40:W * 5 + 40, :] = np.concatenate([resdual_2[..., None], resdual_2[..., None], resdual_2[..., None]], axis=2)
                    
                    cv2.imwrite(dp + '/' + filename[i], rtn)
        print('Val: G:%.8f P: %4.2f -> %4.2f   S: %5.4f -> %5.4f   M: %5.2f -> %5.2f' % (np.mean(gen_losses), np.mean(psnr_o), np.mean(psnr), np.mean(ssim_o), np.mean(ssim), np.mean(mse_o), np.mean(mse)))
        return np.mean(gen_losses), np.mean(psnr), np.mean(ssim), np.mean(mse), np.mean(psnr_o), np.mean(ssim_o), np.mean(mse_o)


    def convert1(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return img

    def convert2(self, x):
        x = x * 255.
        return x.permute(0, 2, 3, 1).cpu().detach().numpy()



def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou


def metric_osn(real, fake):
    psnr, ssim, mse = [], [], []
    for i in range(len(real)):
        psnr.append(peak_signal_noise_ratio(real[i], fake[i], data_range=255))
        ssim.append(structural_similarity(real[i], fake[i], win_size=11, data_range=255.0, multichannel=True))
        mse.append(mean_squared_error(real[i], fake[i]))
    return np.mean(psnr), np.mean(ssim), np.mean(mse)




if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('type', type=str,default='train', required = False,help='train or test the model', choices=['train', 'test', 'val'])
    #parser.add_argument('is_lbp', type=bool, help='if true, only train the lbp network', choices=[True, False])
    #parser.add_argument('is_adv', type=bool, help='if true, train the model with adv', choices=[True, False])
    #parser.add_argument('--is_lbp', type=str, default='no')
    #parser.add_argument('--is_adv', type=str, default='no')
    # parser.add_argument('--type', type=str, default='train')
    
    #args = parser.parse_args()
    
    #if args.type == 'train':
        model = ForgeryForensics()
        model.train()
   
    elif args.type == 'val':
        model = ForgeryForensics()
        model.val()

