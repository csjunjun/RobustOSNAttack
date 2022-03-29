import sys 
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np

from torchvision import transforms

from PIL import Image

from models import ImagenetModels,osn_model
from rattacks import rfgsm
from rattacks import rpgd
from rattacks import rmifgsm
from rattacks import rcw
from rattacks import helper
from utils import getImagenetTrainWithPath

def AttackLinf(imgs,labels,osnmodel,recogmodel,attype,eps,alpha=0.5,random=False,filter=False,steps=None
,cwc=None,lr=None,cwk=None,atttype=0):
    
    advs = None
    adv_p = labels
    flag = 0
    succ = False
    if random == True:
        attype = np.random.randint(4,size=1)[0]

    if attype == 0:
        e = int(eps)/255
        attack1 = rfgsm.RFGSM(recogmodel,osnmodel,eps=e,alpha=alpha)
        advs = attack1(imgs,labels)
        l2 = torch.norm(imgs-advs)
        l1 = torch.norm(imgs-advs,1)
        print("type={},e={},l2={},l1={},alpha={}".format('UNIFGSM',e,l2,l1,alpha,alpha))
        _,adv_p = torch.max(osnmodel(advs),1)

        if torch.any(labels == adv_p):
            succ = False
        else:
            succ = True  
    if attype==1 :
        eps = int(eps)/255
        attack1 = rmifgsm.RMIFGSM(recogmodel,osnmodel,eps=eps,alpha=alpha)
        advs= attack1(imgs,labels)
        l2 = torch.norm(imgs-advs)
        l1 = torch.norm(imgs-advs,1)
        print("type={},e={},l2={},l1={},alpha={}".format('MIFGSM',eps,l2,l1,alpha))
        _,adv_p = torch.max(osnmodel(advs),1)
        #print("target lable:{};predict label:{}".format(target_labels,adv_p))
        if torch.any(labels == adv_p):
            succ = False
        else:
            succ = True
    if attype==2:
        eps = int(eps)/255
        attack1 = rpgd.RPGD(recogmodel,osnmodel,eps=eps,balance=alpha)
        advs= attack1(imgs,labels)
        l2 = torch.norm(imgs-advs)
        l1 = torch.norm(imgs-advs,1)
        print("type={},e={},l2={},l1={},alpha={}".format('UNIPGD',eps,l2,l1,alpha))
        _,adv_p = torch.max(osnmodel(advs),1)
        #print("target lable:{};predict label:{}".format(target_labels,adv_p))
        if torch.any(labels == adv_p):
            succ = False
        else:
            succ = True
    if attype == 3:
        
        c = cwc
        l2 =1000 
        st = steps
        llr = lr
        k =cwk
        attack1 = rcw.RCW(recogmodel,osnmodel,c=c,kappa=k,steps=st,lr=llr,balance=alpha,atttype=atttype) 
        advs = attack1(imgs,labels)
        l2 = torch.norm(imgs-advs)
        l1 = torch.norm(imgs-advs,1)
        print("type={},c={},lr={},steps={},l2={},alpha={},k={}".format('UNICW',c,llr,st,l2,alpha,k))
        _,adv_p = torch.max(osnmodel(advs),1)   
        if torch.any(labels == adv_p):
            succ = False
        else:
            succ = True
    if filter == True:  
        if advs is None or torch.any(adv_p==labels):
            advs = 0  
    else:
        if advs is None:
            advs=0
    return advs,int(adv_p),attype,succ


if __name__ =="__main__":
    img_size = 256
    imgnet_testloader = getImagenetTrainWithPath('/home/junliu/data/ImageNet_val',1,resize=True,isshuffle=False,resize_v=img_size)
    fail = 0 
    suc = 0
    recog_suc = 0
    model_fail = 0
    model_suc = 0
    diffent =0
    #recgmodel = ImagenetModels.IncepV3_Imagenet()
    recgmodel = ImagenetModels.Resnet50_Imagenet()
    #recgmodel = ImagenetModels.VGG19bn_Imagenet()
    osnmodel = osn_model.OSN_Model(recgmodel)
    print("predict main :{}".format(recgmodel.name))
    img_num = 0
    x_advs,adv_labels,true_labels,randattacks,diff_recovers,diff_advs =[],[],[],[],[],[]
    transform_pil_tensor = transforms.Compose([
	transforms.ToTensor()
	]
)
 
    for items in imgnet_testloader:
        if model_suc==200:
            break
        x, y,paths = items ###TODO
        x    = x.cuda()
        y    = y.cuda()
        x_qf = int(helper.jpeg_qtableinv(paths[0]))
        print("shape========:{}".format(x.shape))
        print("max size:{}".format(max(x.shape[-2],x.shape[-1])))

        output = recgmodel(x)
        _,preds = torch.max(output,1)
        if torch.any(preds != y):
            model_fail += 1
            print("recgmodel predict wrong")
            img_num += 1
            continue
        model_suc+=1
        attype = 0
        eps = 3
        st=100
        alpha =0.3
        cwc =2
        lr = 0.01
        cwk=0
        x_adv,adv_label,attype,succ = AttackLinf(x,y,osnmodel,recgmodel,attype,eps,alpha=alpha,random=False,filter=False,steps=st,cwc=cwc,lr=lr,cwk=cwk,atttype=1)
       
        if x_adv is not None and x_adv.__class__.__name__ == 'Tensor':
            if succ :
                succ_f = 1
                suc += len(torch.where(adv_label!=y)[0])
                print("attack {} success".format(img_num))
            else:
                succ_f = 0
                fail += 1
                print("attack {} fail".format(img_num))
        

            _,adv_recog = torch.max(recgmodel(x_adv),1)
            if torch.all(adv_recog!=y):
                vallina_suc =1
                recog_suc +=1
            else:
                vallina_suc =0

            adv_img = Image.fromarray(np.asarray(x_adv[0].cpu().permute(1,2,0).numpy()*255,dtype=np.uint8))

            dp = 'advimgs/osn_rfgsm_imgnetval256_ep{}_alpha0@{}'.format(eps,int(alpha*10)) # corres
           
            if not os.path.exists(dp):
                os.mkdir(dp)
                print("new folder {}".format(dp))
            imgpath = dp+'/{}_attype{}_true{}_adv{}_qf{}_vallinasuc{}_osnsuc{}.png'.format(img_num,attype,int(y.cpu()),adv_label,x_qf,vallina_suc,succ_f)
            adv_img.save(imgpath)
            print(imgpath)
           
            diff_adv = torch.norm(x_adv-x)
            print("advs l2 = {}".format(diff_adv))
            diff_advs.append(float(diff_adv.cpu()))            
            adv_labels.append(adv_label)
            true_labels.append(int(y.cpu()))

        else:
            print("attack {} fail outer".format(img_num))
            fail += 1 
        if model_suc%50 == 0:
            print("attack osnmodel success rate:{}".format(suc/model_suc))
            print("model success rate:{}".format(1-(model_fail/(model_suc+model_fail))))
            print("different save adv:{}".format(diffent))
            print("mean diff_recovers:{}".format(np.mean(np.asarray(diff_recovers))))
            print("mean diff_advs ={}".format(np.mean(np.asarray(diff_advs))))
            print("attack vallina model success rate:{}".format(recog_suc/model_suc))

        img_num += 1
    print("attack osnmodel success rate:{}".format(suc/model_suc))
    print("model success rate:{}".format(1-(model_fail/(model_suc+model_fail))))
    print("different save adv:{}".format(diffent))
    print("mean diff_recovers:{}".format(np.mean(np.asarray(diff_recovers))))
    print("mean diff_advs ={}".format(np.mean(np.asarray(diff_advs))))
    print("attack vallina model success rate:{}".format(recog_suc/model_suc))
   
    print("finished")


