from forward_model import transfer_matrix as tmm
from utils import *

if __name__ == '__main__':
    gamma = 0.99
    spec_dim = 401
    num = 401
    wlmin = 400
    wlmax = 800
    mat_lst = ['air', 'SiO2', 'Al2O3', 'SiC', 'HfO2', 'SiO2', 'TiO2', 'Ag', 'Al2O3', 'HfO2', 'SiC', 'HfO2', 'glass']
    tm = tmm(mat_lst= mat_lst,
             num=num, wlmin=wlmin, wlmax=wlmax)
    target = tm.target
    wl = tm.wl
    lst = []
    x0 = torch.tensor([1.1539e+02, 2.9681e+02, 1.3400e+02, 3.0299e+02, 3.4010e+02, 1.6454e+02,
        1.2408e-01, 3.0179e+02, 1.5700e+02, 1.0976e+01, 1.6815e+02])
    lst.append(x0)
    #x0 = torch.tensor([88., 67, 115, 140, 21, 188, 88, 151, 48])
    #lst.append(x0)
    #lab = ['GA optim', 'DL optim']
    lab = ['DL optim']
    plt.rcParams['font.size'] = 15
    plt.figure(dpi=300)
    ep = 2000
    for idx, x0 in enumerate(lst):
        lr = 0.01
        print('idx = {}'.format(idx))
        x = torch.nn.parameter.Parameter(x0, requires_grad=True)
        err = torch.zeros(ep)
        for i in range(ep):
            y = tm(x)
            loss = tm.loss_function3(y,x)
            err[i] = loss.item()
            grad = torch.autograd.grad(loss, x)[0]
            x = x - grad * lr
            relu = torch.nn.ReLU()
            x = relu(x)
            lr = lr * gamma

            # print(x)
        plt.plot(torch.arange(ep), err, label=lab[idx])
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.figure(dpi=400)
        plt.plot(wl, target, label='target')
        y = tm(x).detach().cpu()
        plt.plot(wl, y, label='DL desgin')
        print("transmittance is {}".format(y))
        plt.xlabel('wavelength (nm)')
        plt.ylabel('transmittance')
        # plt.ylim(0, 0.023)
        print(x, err[-1])
    plt.legend()
    plt.show()