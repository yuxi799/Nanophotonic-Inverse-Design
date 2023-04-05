from __future__ import print_function
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
import argparse
from utils import *
from forward_model import transfer_matrix as tmm
from inverse_model import inverse_model as invmodel
from AutomaticWeightedLoss import AutomaticWeightedLoss

parser = argparse.ArgumentParser(description='training tandem net for multilayer design')
# Parameters for  training
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--es', default=20, type=int, help='epoch size')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--bs', default=5, type=int, help='batch size, better to have a square number')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--scheduler_gamma', default=0.97, type=float, help='weight decay')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model, ensuring the resume path is given')
parser.add_argument('--val_num', default=8, type=int, help=' the number of validate images, also nrow for saving images')
args = parser.parse_args()


def main():
    start_epoch = 0
    best_loss = 9999999.99

    # Model
    print('==> Building model..')
    net = invmodel(spec_dim=spec_dim, layer_dim=layer_dim, hidden_dims=[512,64,32,16])
    net = net.to(device)



    if device == 'cuda':
        # Considering the data scale and model, it is unnecessary to use DistributedDataParallel
        # which could speed up the training and inference compared to DataParallel
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    awl = AutomaticWeightedLoss(2)
    optimizer = optim.Adam([{'params': awl.parameters()},{'params': net.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            #logger = Logger(os.path.join(args.checkpoint, 'log_na.txt'), resume=True)
            logger = Logger(os.path.join(args.checkpoint, 'log_na.txt'))
            logger.set_names(['Epoch', 'LR', 'Train Loss'])
            #print('==> Resuming from checkpoint, loaded..')
        else:
            print("==> No checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_na.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    if not args.evaluate:
        # training
        print("==> start training..")
        for epoch in range(start_epoch, start_epoch + args.es):
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, scheduler.get_last_lr()[-1]))
            train_out = train(net, optimizer,awl)  # {train_loss, amp_loss, cos_loss, phi_loss}
            test_out = test(net,awl)
            save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint_na.pth'))
            if test_out["test_loss"] < best_loss:
                save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint_na_best.pth'),
                           loss=test_out["test_loss"])
                best_loss = test_out["test_loss"]
            logger.append([epoch + 1, scheduler.get_last_lr()[-1], train_out["train_loss"]])
            logger.append([epoch + 1, scheduler.get_last_lr()[-1], test_out["test_loss"]])
            scheduler.step()
        logger.close()
        print(f"\n==> Finish training..\n")
    return net
    # print("===> start evaluating ...")
    # pred_spectrum(net, valloader, name="test_reconstruct")


def train(net, optimizer, awl, batch=10,):
    net.train()
    train_loss = 0
    target = tm.target.to(device)
    #print('specs是否是叶张量：', specs.is_leaf)
    for idx in range(batch):
        #layers, specs = tm.generate_data(args.bs)
        #layers = layers.to(device)
        #specs = specs.to(device)
        specs = target.unsqueeze(0).repeat(args.bs, 1).to(device)
        #specs_1 = torch.nn.parameter.Parameter(specs, requires_grad=True)
        optimizer.zero_grad()
        #print('specs grad of loss before is {}'.format(specs.grad))
        recons = net(specs)
        recons = recons * (tmax - tmin) + tmin
        preds,_ = tm(recons)
        _, preds_var = tm(recons)
        #awl = AutomaticWeightedLoss(2)
        loss1 = tm.loss_function1(preds)  # loss, Reconstruction_Loss, KLD
        loss2 = tm.loss_function4(preds_var) * 50
        #print(loss1,loss2)
        # add loss 2
        # invdes = net(target.unsqueeze(0).repeat(2, 1))
        # invdes = invdes * (tmax - tmin) + tmin
        # pre = tm(invdes)[0]
        # loss2 = torch.mean((pre - target) ** 2)

        # loss = loss
        loss_sum = awl(loss1,loss2)
        loss1.backward()
        #loss_sum.backward()
        optimizer.step()
        #print(specs)
        #print('specs grad is {}'.format(specs_1.grad))
        #print('lr is {}'.format(args.lr))
        #specs_1 = specs_1 - np.float64(args.lr) * specs_1.grad
        #print('specs_1 after neural adjoint is {}'.format(specs_1))
        #print('specs是否是叶张量：', specs.is_leaf)
        train_loss += loss1.item()
        #train_loss += loss_sum.item()
        print('training mse for idx {} is {}'.format(idx, loss1))
        # print('inverse design mse for idx {} is {}'.format(idx, loss2))
        # progress_bar(idx, 'Loss: %.3f' % (train_loss / (idx + 1)))
    # plot
    wl = tm.wl
    #opt, _, _, _ = tm.cal_trans(x0)
    invdes = net(target.unsqueeze(0).repeat(2, 1).to(device))
    invdes = invdes * (tmax - tmin) + tmin
    print(invdes[0])
    tm.plot_config(invdes[0].detach().cpu(), save='results/design_ep = {}.png'.format(ep), tmin=tmin, tmax=tmax)
    #pre = tm(invdes)[0].detach().cpu()
    pre,_ = tm.cal_trans(invdes[0].detach().cpu(),plot = True)
    #target = target.cpu()
    #err_opt = tm.loss_function(opt)
    #err_pre = tm.loss_function1(pre)
    #plt.figure(dpi=300)
    #plt.plot(wl, target, label='target')
    #plt.plot(wl, opt, label='optim design')
    #plt.plot(wl, pre, label='DL design')
    #plt.xlabel('wavelength (nm)')
    #plt.ylabel('transmittance')
    #plt.title('pred err {:.5f}'.format(err_pre))
    #plt.legend()
    #plt.savefig('results/spec_ep = {}.png'.format(ep))
    #plt.show()

    return {"train_loss": train_loss / (idx + 1)}


def test(net,awl):##为啥会过拟合
    global ep
    ep = ep + 1
    net.eval()
    test_loss = 0
    idx = 0
    target = tm.target
    #layers, specs = tm.generate_data(args.bs)
    #layers = layers.to(device)
    #specs = specs.to(device)
    specs = target.unsqueeze(0).repeat(args.bs, 1).to(device)
    with torch.no_grad():
        recons = net(specs)
        recons = recons * (tmax - tmin) + tmin
        preds,_ = tm(recons)
        _,preds_var = tm(recons)
        loss1 = tm.loss_function1(preds)  # loss, Reconstruction_Loss, KLD
        loss2 = tm.loss_function4(preds_var)
        #awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1,loss2)
    test_loss += loss_sum.item()
    #  plot
    #wl = tm.wl
    #opt, _, _, _ = tm.cal_trans(x0)
    #invdes = net(target.unsqueeze(0).repeat(2, 1).to(device))
    #invdes = invdes * (tmax - tmin) + tmin
    #tm.plot_config(invdes[0], save='results/design_ep = {}.png'.format(ep))
    #pre = tm(invdes)[0].detach().cpu()
    # err_opt = torch.mean((opt - target) ** 2)
    #err_pre = torch.mean((pre - target) ** 2)
    #plt.figure(dpi=300)
    #plt.plot(wl, target, label='target')
    # plt.plot(wl, opt, label='optim design')
    #plt.plot(wl, pre, label='DL design')
    #plt.xlabel('wavelength (nm)')
    #plt.ylabel('transmittance')
    #plt.title('pred err {}'.format(err_pre))
    #plt.legend()
    #plt.savefig('results/spec_ep = {}.png'.format(ep))
    #plt.show()
    #progress_bar(idx, 'Loss: %.3f' % (test_loss / (idx + 1)))
    print('test mse for idx {:.5f} is {:.5f}'.format(idx, loss_sum))
    return {"test_loss": test_loss / (idx + 1)}


if __name__ == '__main__':
    global ep
    layer_dim = 11
    ep = 0
    test_mode = True
    args.small = False
    if test_mode:
        args.small = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.checkpoint = './checkpoints/{}_{}layer'.format('tandem_all', layer_dim)
    if not os.path.isdir(args.checkpoint):
        os.mkdir(args.checkpoint)
    # data loader
    spec_dim = 401
    num = 401
    wlmin = 400
    wlmax = 800
    #t = '235	134	215	185	124	142	254	304	170	130	13	74	13	165	92	92	84	274	70	64	183	138	252	58	148	161'
    #x0 = [float(x) for x in t.split()]
    #x0 = torch.tensor(x0)
    x0 = torch.tensor([ 56, 134, 130, 187, 151,  70,  15, 198, 107, 193,  82])
    d = 50
    tmin = torch.maximum(x0 - d, torch.zeros(layer_dim)).to(device)
    tmax = (x0 + d).to(device)
    # x0 = torch.tensor([91.1, 51.6, 190.3, 19.8, 154.1, 109.3])
    #str = 'SiO2	Al2O3	HfO2	Al2O3	Al2O3	MgF2	TiO2	MgF2	MgF2	TiO2	SiC	ITO	Ag	SiO2	TiO2	MgF2	SiC	Al2O3	SiO2	HfO2	TiO2	MgF2	SiC	MgF2	Al2O3	MgF2'
    #mat_lst = ['air'] + str.split() + ['air']
    mat_lst =['air', 'ITO', 'SiO2', 'SiN', 'HfO2', 'MgF2', 'HfO2', 'Ag', 'SiC', 'HfO2', 'HfO2', 'SiN', 'glass']
    tm = tmm(mat_lst=mat_lst, num=num, wlmin=wlmin, wlmax=wlmax)
    net = main()
