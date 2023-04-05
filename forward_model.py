#穿过multiplayer后有target transmission spectrum,the design space is the thickness of each layer
import torch
import torch.nn as nn
from utils import *
from matplotlib.patches import Rectangle, Circle



class transfer_matrix(nn.Module):
    def __init__(self, mat_lst=['air', 'TiO2', 'MgF2', 'TiO2', 'Ag', 'MgF2', 'TiO2', 'SiO2'],
                 wlmin=400, wlmax=800, num=401, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        initialize the transfer matrix class
        :param mat_lst: a list of materials, include the semi-inf air and include the semi-inf sub
        :param wlmin: min wavelength, unit: nm
        :param wlmax: max wavelength, unit: nm
        :param num: wavelength data point number
        """
        super(transfer_matrix, self).__init__()
        self.mat_lst = mat_lst  # layer material list
        self.n_mat = len(mat_lst)  # number of layers
        self.wlmin = wlmin
        self.wlmax = wlmax
        self.num = num
        self.wl = torch.linspace(wlmin, wlmax, num)  # wavelength list (start,end,point number)
        self.mat_n0, self.mat_k0 = self.get_index_spec()  # get n0 and k0 for all wavelengths and all layers
        self.target = target_trans(wlmin=wlmin, wlmax=wlmax, num=num, plot=False)
        self.solar = solar(wlmin=wlmin, wlmax=wlmax, num=num, plot=False)

    def get_index_spec(self):
        """
        get the refractive index matrix, row for different wavelength, col for different layer
        :return: two (real / imag) index matrix with dimension [num, len(mat_lst)]
        """
        n_mat = len(self.mat_lst) # numbers of materials
        mat_n0 = torch.ones(self.num, n_mat) # assigned the value 1
        mat_k0 = torch.zeros(self.num, n_mat)#
        for i, mat in enumerate(self.mat_lst):
            _, n0, k0 = get_index(mat, wlmin=self.wlmin, wlmax=self.wlmax, num=self.num, plot=False)  # get ind for i-th layer, [num]
            mat_n0[:, i] = n0 #不同波长下的不同的N,K
            mat_k0[:, i] = k0
        mat_k0[:, -1] = 0  # force the last layer to be lossless？
        return mat_n0, mat_k0  # [num, n_mat)]

    def cal_trans(self, x, plot=False): #calculate actual T与厚度有关
        """
        calculate the transmittance of the given multilayer, the 0-th layer is air by default with n = 1
        :param x: a list of thickness parameters, without the first air layer
        :return:
        """
        # notice air layer and last layer are not included in x, so x[i] is the thickness of i+1-th layer
        device = x.device.type
        n_mat = len(self.mat_lst)
        #print(len(x))
        assert len(x) == n_mat - 2, 'number of thickness does not match the materials'
        # 写个theta_i的数组，动态数组,保存初始值
        # theta_i = np.pi / 3
        T_s_total = torch.Tensor(0)
        T_p_total = torch.Tensor(0)
        T_total = torch.Tensor(0)
        angle_max = 60
        theta_i = []# try 10 degree difference
        for i in range(0,angle_max + 1):
            theta_i = theta_i + [[i]*self.num]
        #print(theta_i)
        theta_i = torch.from_numpy(np.pi * np.array(theta_i) / 180).type(torch.complex128)
        # theta_i = 0
        #theta_b = torch.real(theta_i)
        #print(theta_b)
        for ang in range(len(theta_i)):
            # the initial transfer matrix   [num, 2, 2]
            smat_s = torch.eye(2).unsqueeze(0)
            smat_s = smat_s.repeat(self.num, 1, 1).type(torch.complex128).to(device) #[num,2,2]
            smat_p = torch.eye(2).unsqueeze(0)
            smat_p = smat_p.repeat(self.num, 1, 1).type(torch.complex128).to(device)
            # iterate to calculate Iij * Lj from I01 * L1 to In-3,n-2 * Ln-2 and In-2,n-1
            for i in range(n_mat - 1):
                j = i + 1
                ni = torch.complex(self.mat_n0[:, i], self.mat_k0[:, i]).to(device)  # i-th layer index, [num]
                nj = torch.complex(self.mat_n0[:, j], self.mat_k0[:, j]).to(device)  # j-th layer index, [num]
                #theta_i, theta_t = nj * theta_i / ni
                theta_t = np.arcsin(np.sin(theta_i[ang]) * ni / nj)
                #print('ni is {},nj is {}'.format(ni,nj))
                #theta_t保存在theta_i里面
                #rij = (ni - nj) / (ni + nj)  # reflection coef
                #tij = 2 * ni / (ni + nj)  # transmission coef
                #theta_i初始为60，算出theta_t,然后把theta_t保存至theta_i,然后提取最新的theta_i.再算theta_t.
                rij_s = (ni * np.cos(theta_i[ang]) - nj * np.cos(theta_t))/(ni * np.cos(theta_i[ang]) + nj * np.cos(theta_t))
                tij_s = 2 * ni * np.cos(theta_i[ang]) / ( ni * np.cos(theta_i[ang]) + nj * np.cos(theta_t))
                rij_p = (ni * np.cos(theta_t) - nj * np.cos(theta_i[ang])) / (ni * np.cos(theta_t) + nj * np.cos(theta_i[ang]))
                tij_p = 2 * ni * np.cos(theta_t) / (ni * np.cos(theta_t) + nj * np.cos(theta_i[ang]))
                #rij = (rij_s + rij_p) / 2
                #tij = (tij_s + tij_p) / 2
                #print('theta_i is {}, theta_t is {}'.format(theta_i, theta_t))
                theta_i[ang] = theta_t
                # the Iij and Lj matrices are both with shape [num, 2, 2]
                Iij_s = torch.stack((torch.stack((torch.ones(self.num).to(device), rij_s), dim=1),
                                   torch.stack((rij_s, torch.ones(self.num).to(device)), dim=1)), dim=2) \
                      / tij_s.unsqueeze(1).unsqueeze(1).type(torch.complex128) #?[[[1,r12],[r12,1]],[[1,r23],[r23,1],...]
                smat_s = smat_s.matmul(Iij_s) #?[[[1,r12],[r12,1]],[[1,r23],[r23,1]],...]
                if i < n_mat - 2: #?In-2,n-1. should be In-3,n-2
                    Lj_s = torch.stack((torch.stack((torch.exp(-1j * 2 * np.pi * nj * x[i]*np.cos(theta_t) / self.wl.to(device)),
                                                   torch.zeros(self.num).to(device)), dim=1),
                                      torch.stack((torch.zeros(self.num).to(device),
                                                   torch.exp(1j * 2 * np.pi * nj * x[i]*np.cos(theta_t) / self.wl.to(device))),
                                                  dim=1)), dim=2).type(torch.complex128) #[[[-1j * 2 * np.pi * nj * x[i],0],[0,1j * 2 * np.pi * nj * x[i]]]]
                    smat_s = smat_s.matmul(Lj_s) #[[[-j...,jr12],[-jr12,j...],...]]
                Iij_p = torch.stack((torch.stack((torch.ones(self.num).to(device), rij_p), dim=1),
                                     torch.stack((rij_p, torch.ones(self.num).to(device)), dim=1)), dim=2) \
                        / tij_p.unsqueeze(1).unsqueeze(1).type(torch.complex128)  # ?[[[1,r12],[r12,1]],[[1,r23],[r23,1],...]
                smat_p = smat_p.matmul(Iij_p)  # ?[[[1/t,r12/t],[r12/t,1/t]],[[1,r23],[r23,1]],...]
                if i < n_mat - 2:  # ?In-2,n-1. should be In-3,n-2
                    Lj_p = torch.stack(
                        (torch.stack((torch.exp(-1j * 2 * np.pi * nj * x[i] * np.cos(theta_t) / self.wl.to(device)),
                                      torch.zeros(self.num).to(device)), dim=1),
                         torch.stack((torch.zeros(self.num).to(device),
                                      torch.exp(1j * 2 * np.pi * nj * x[i] * np.cos(theta_t) / self.wl.to(device))),
                                     dim=1)), dim=2).type(torch.complex128)  # [[[-1j * 2 * np.pi * nj * x[i],0],[0,1j * 2 * np.pi * nj * x[i]]]]
                    smat_p = smat_p.matmul(Lj_p)  # [[[-j...,jr12],[-jr12,j...],...]]
            # extract refl and trans coefficient and intensity
            r_s = smat_s[:, 1, 0] / smat_s[:, 0, 0]#[rij/2pi*nj*x[i]]
            t_s = 1 / smat_s[:, 0, 0] #[1/-j...,]
            R_s = r_s.abs() ** 2
            T_s = t_s.abs() ** 2 * (nj * np.cos(theta_t)).real / (self.mat_n0[:,0] * np.cos(np.pi * ang / 180).real).to(device)
            #print(theta_b)
            #print(theta_b.type,ang,theta_b[ang])
            #print(1,theta_b)
            r_p = smat_p[:, 1, 0] / smat_p[:, 0, 0]  # [rij/2pi*nj*x[i]]
            t_p = 1 / smat_p[:, 0, 0]  # [1/-j...,]
            R_p = r_p.abs() ** 2
            T_p = t_p.abs() ** 2 * ((nj / np.cos(theta_t)).real / (self.mat_n0[:,0] / np.cos(np.pi * ang / 180)).real).to(device)
            T = (T_s + T_p) / 2
            T_s = T_s.unsqueeze(0)
            T_p = T_p.unsqueeze(0)
            T = T.unsqueeze(0)
            T_s_total = torch.cat((T_s_total,T_s),dim = 0)
            T_p_total = torch.cat((T_p_total,T_p),dim = 0)
            T_total = torch.cat((T_total,T),dim = 0)
        #print(T_s.squeeze(0))
        #print(T_s_total[0])
        T_s_total_mean = torch.mean(T_s_total,dim = 0 ,keepdim=True)
        T_p_total_mean = torch.mean(T_p_total,dim = 0,keepdim=True)
        T_total_mean = torch.mean(T_total,dim=0,keepdim=True)
        T_s_total_var = torch.var(T_s_total,dim = 0,keepdim=True)
        T_p_total_var = torch.var(T_p_total, dim=0,keepdim=True)
        T_total_var = torch.var(T_total, dim=0,keepdim=True)
        #print(T_s_total_mean.shape)
        #print(T_s_total_var.shape)
        if plot:
            T = T.cpu()
            plt.plot(self.wl, T_s_total[-1], label='transmittance_s_60')
            #plt.plot(self.wl, T_p_total[60],label='transmittance_p_60')
            plt.plot(self.wl, T_s_total[0], label = 'transmittance_0')
            plt.plot(self.wl, T_s_total_mean[0], label='transmittance_mean')
            plt.plot(self.wl, T_s_total_var[0], label = 'var')
            # plt.plot(self.wl, R, label='reflectance')
            plt.plot(self.wl, self.target, label='target')
            plt.xlabel('wavelength (nm)')
            plt.ylabel('trans/refl')
            plt.legend()
            plt.show()
        return T_s_total_mean[0], T_s_total_var[0]#?r,t,R,T
        #return T_s_total_mean[0], T_s_total_var[0],T_s_total[0],T_s_total[60]
        #return T_s_total[0]

    def cal_trans_batch(self, x, plot=False): #according to batch, calculate the actual T
        """
        calculate the transmittance of a batch of given multilayer, the 0-th layer is air by default with n = 1
        :param x: a list of thickness parameters, without the first air layer and last layer [-1, len(mat_lst) - 2]
        :return:
        """
        # notice air layer and last layer are not included in x, so x[i] is the thickness of i+1-th layer
        bs = len(x)  # batch size
        device = x.device.type
        n_mat = len(self.mat_lst)
        assert len(x[0]) == n_mat - 2, 'number of thickness does not match the materials, one is {} the other is {}'.format(len(x[0]), n_mat-2)
        wl = self.wl.unsqueeze(1).repeat(1, bs).to(device)
        # the initial transfer matrix   [num, 2, 2]
        T_s_total = torch.Tensor(0)
        T_p_total = torch.Tensor(0)
        T_total = torch.Tensor(0)
        angle_max = 60
        theta_i = []
        for i in range(angle_max + 1):
            theta_i = theta_i + [[i] * self.num]
        theta_i = torch.from_numpy(np.pi * np.array(theta_i) / 180).type(torch.complex128)
        for ang in range(len(theta_i)):
            smat_s = torch.eye(2).unsqueeze(0).unsqueeze(0) #[2,2] - [1,1,2,2]
            smat_s = smat_s.repeat(bs, self.num, 1, 1).type(torch.complex128).to(device) #[bs,num,2,2]
            smat_p = torch.eye(2).unsqueeze(0).unsqueeze(0)  # [2,2] - [1,1,2,2]
            smat_p = smat_p.repeat(bs, self.num, 1, 1).type(torch.complex128).to(device)  # [bs,num,2,2]

            # iterate to calculate Iij * Lj from I01 * L1 to In-3,n-2 * Ln-2 and In-2,n-1
            #theta_i = np.pi / 3
            #theta_a = theta_i * torch.ones(401)
            #theta_i = 0
            for i in range(n_mat - 1):
                j = i + 1
                ni = torch.complex(self.mat_n0[:, i], self.mat_k0[:, i]).to(device)  # i-th layer index, [num]
                nj = torch.complex(self.mat_n0[:, j], self.mat_k0[:, j]).to(device)  # j-th layer index, [num]
                #
                #rij = (ni - nj) / (ni + nj)  # reflection coef
                #tij = 2 * ni / (ni + nj)  # transmission coef
                theta_t = np.arcsin(np.sin(theta_i[ang]) * ni / nj)
                #
                rij_s = (ni * np.cos(theta_i[ang]) - nj * np.cos(theta_t)) / (ni * np.cos(theta_i[ang]) + nj * np.cos(theta_t))
                tij_s = 2 * ni * np.cos(theta_i[ang]) / (ni * np.cos(theta_i[ang]) + nj * np.cos(theta_t))
                rij_p = (ni * np.cos(theta_t) - nj * np.cos(theta_i[ang])) / (ni * np.cos(theta_t) + nj * np.cos(theta_i[ang]))
                tij_p = 2 * ni * np.cos(theta_t) / (ni * np.cos(theta_t) + nj * np.cos(theta_i[ang]))
                #rij = rij_p
                #tij = tij_p
                #rij = (rij_s + rij_p) / 2
                #tij = (tij_s + tij_p) / 2
                theta_i[ang] = theta_t
                # the Iij and Lj matrices are both with shape [bs, num, 2, 2]
                Iij_s = torch.stack((torch.stack((torch.ones(self.num).to(device), rij_s), dim=1),
                                   torch.stack((rij_s, torch.ones(self.num).to(device)), dim=1)), dim=2) \
                      / tij_s.unsqueeze(1).unsqueeze(1).type(torch.complex128)
                Iij_s = Iij_s.unsqueeze(0).repeat(bs, 1, 1, 1)
                smat_s = smat_s.matmul(Iij_s)
                nj = nj.unsqueeze(1).repeat(1, bs)
                theta_t = theta_t.unsqueeze(1).repeat(1,bs)
                if i < n_mat - 2:
                    Lj_s = torch.stack((torch.stack((torch.exp(-1j * 2 * np.pi * nj * x[:, i]*np.cos(theta_t) / wl).transpose(0, 1),
                                                   torch.zeros(bs, self.num).to(device)), dim=2),
                                      torch.stack((torch.zeros(bs, self.num).to(device),
                                                   torch.exp(1j * 2 * np.pi * nj * x[:, i]*np.cos(theta_t) / wl).transpose(0, 1)),
                                                  dim=2)), dim=3).type(torch.complex128)
                    smat_s = smat_s.matmul(Lj_s)
                Iij_p = torch.stack((torch.stack((torch.ones(self.num).to(device), rij_p), dim=1),
                                     torch.stack((rij_p, torch.ones(self.num).to(device)), dim=1)), dim=2) \
                        / tij_p.unsqueeze(1).unsqueeze(1).type(torch.complex128)
                Iij_p = Iij_p.unsqueeze(0).repeat(bs, 1, 1, 1)
                smat_p = smat_p.matmul(Iij_p)
                if i < n_mat - 2:
                    Lj_p = torch.stack(
                        (torch.stack((torch.exp(-1j * 2 * np.pi * nj * x[:, i] * np.cos(theta_t) / wl).transpose(0, 1),
                                      torch.zeros(bs, self.num).to(device)), dim=2),
                         torch.stack((torch.zeros(bs, self.num).to(device),
                                      torch.exp(1j * 2 * np.pi * nj * x[:, i] * np.cos(theta_t) / wl).transpose(0, 1)),
                                     dim=2)), dim=3).type(torch.complex128)
                    smat_p = smat_p.matmul(Lj_p)
            # extract refl and trans coefficient and intensity
            r_s = smat_s[:, :, 1, 0] / smat_s[:, :, 0, 0]
            t_s = 1 / smat_s[:, :, 0, 0]
            R_s = r_s.abs() ** 2
            T_s = t_s.abs() ** 2 * (nj[:,0] * np.cos(theta_t[:,0])).real/ (self.mat_n0[:,0] * np.cos(np.pi * ang * torch.ones(self.num)/ 180)).real.to(device)  # [bs, num]
            r_p = smat_p[:, :, 1, 0] / smat_p[:, :, 0, 0]
            t_p = 1 / smat_p[:, :, 0, 0]
            R_p = r_p.abs() ** 2
            T_p = t_p.abs() ** 2 * (nj[:,0] / np.cos(theta_t[:,0])).real / (self.mat_n0[:,0] / np.cos(np.pi * ang *torch.ones(self.num)/ 180)).real.to(device)  # [bs, num]
            T = (T_s + T_p) / 2
            T_s = T_s.unsqueeze(0)#[1,bs,num]
            T_p = T_p.unsqueeze(0)
            T = T.unsqueeze(0)
            #print(T_s.shape)
            T_s_total = torch.cat((T_s_total,T_s),dim = 0)
            T_p_total = torch.cat((T_p_total,T_p),dim = 0)
            T_total = torch.cat((T_total,T),dim = 0)
            T_s_total_mean = torch.mean(T_s_total, dim=0, keepdim=True)
            T_p_total_mean = torch.mean(T_p_total, dim=0, keepdim=True)
            T_total_mean = torch.mean(T_total, dim=0, keepdim=True)
            T_s_total_var = torch.var(T_s_total, dim = 0, keepdim=True)
            T_p_total_var = torch.var(T_p_total, dim = 0, keepdim=True)
            T_total_var = torch.var(T_total, dim = 0, keepdim=True)
            #print(T_s_total_mean.shape,T_s_total_mean[0].shape,T_s_total_mean[0][0].shape)
        if plot:
            plt.plot(self.wl, T_s.squeeze(0), label='transmittance')
            plt.plot(self.wl, T_s_total[0][0], label = 'transmittance_0')
            plt.plot(self.wl, T_s_total_mean[0][0].cpu(), label='transmittance')
            # plt.plot(self.wl, R, label='reflectance')
            plt.plot(self.wl, self.target, label='target')
            plt.xlabel('wavelength (nm)')
            plt.ylabel('trans/refl')
            plt.legend()
            plt.show()
        #print(T_s_total.shape)
        #print(T_s_total_mean.shape)
        return T_s_total_mean[0], T_s_total_var[0]
        #return T_s_total[0]

    def plot_config(self, x, save=None, tmin=None, tmax=None):
        """
        this function plot the design configuration with the layer thickness list x
        :param x: input thickness list
        :return: plot a figure of the structure
        """
        x = x.detach()
        a = 0.8  # color alpha
        mat_lst = self.mat_lst[1:]  # ignore air layer
        mat_lst.reverse()
        # color for each material
        cmap = {'TiO2': [0.6, 0.6, 0.6, a], 'MgF2': [0.7, 0.6, 0.55, a],
                'Ag': [0.72, 0.75, 0.53, a], 'SiO2': [0.57, 0.66, 0.71, a],
                'HfO2': [0, 0.56, 0.45, a], 'SiC': [0.84, 0.41, 0, a],
                'ITO': [0.98, 0.23, 0.18, a], 'Al2O3': [0.46, 0.17, 0.53, a],
                'SiN': [0.14, 0.22, 0.29, a], 'Si': [0.2, 0.2, 0.2, a], 'air': [1, 1, 1, 1],
                'AZO':[0.3,0.7,0.6,a],'glass':[0.4,0.8,0.7,a],'Y2O3':[0.5,0.5,0.5,a],'HfO2_1':[0.52,0.78,0.69,a],
                'MgF2Notch532':[0.3,0.2,0.7,a],'GlassNotch532':[0.4,0.8,0.2,a],'metafilmNotch532':[0.6,0.9,0.1,a]}
        left = 1  # x_left
        h = 0  # initial height, change when adding material layer
        w = 4  # width of each layer
        hsub = 2  # height of substrate
        ratio = 100  # scale
        xf = x.flip(0)  ##垂直翻转？一维数组怎么flip
        if tmin is not None:
            tminf = tmin.flip(0)
            tmaxf = tmax.flip(0)
        t = xf / ratio  # thickness of each layer in the plot
        fig, ax = plt.subplots(dpi=300)#画图?dpi
        patches = []
        # add material caption
        for i, key in enumerate(cmap.keys()):
            cir = Circle((6, 6 - i), radius=0.3, facecolor=cmap[key])  #为什么画圆？
            ax.text(6.5, 5.9 - i, key)#显示材料
            patches.append(cir)
        # add substrate
        sub = Rectangle((left, h), w, hsub, facecolor=cmap[mat_lst[0]])
        h += hsub
        patches.append(sub)
        # add other layers
        for i in range(1, len(mat_lst)):
            # print(i, mat_lst[i], t[i-1].item(), h)
            rect = Rectangle((left, h), w, t[i - 1], facecolor=cmap[mat_lst[i]])
            # add text
            color = 'k'
            if tmin is not None:#为什么限制厚度？
                tmaxf = tmaxf.cpu()
                tminf = tminf.cpu()
                if (xf[i - 1] + 5 < tmaxf[i - 1] and xf[i - 1] - 5 > tminf[i - 1]):
                    color = 'k'
                elif xf[i - 1] + 5 > tmaxf[i - 1]:
                    color = 'r'
                else:
                    color = 'b'
            ax.text(-3, h + t[i - 1] / 2 - 0.1, '{:.1f} nm'.format(xf[i - 1].item()), color=color)#显示厚度
            h = h + t[i - 1]
            patches.append(rect)
        # plot all shapes
        for shape in patches:
            ax.add_patch(shape)
        plt.box('off')
        plt.axis('off')
        plt.axis('equal')
        if save is not None:
            plt.savefig(save)
        plt.show()

    def forward(self, x):
        if len(x.size()) == 1:
            spec,spec_var = self.cal_trans(x, plot=False)
            #spec = self.cal_trans(x, plot=False)
        else:
            spec, spec_var = self.cal_trans_batch(x, plot=False)
            #spec = self.cal_trans_batch(x)
        return spec,spec_var

    def generate_data(self, n, t_min=50, t_max=200):##产生的什么数据
        """
        generate training data
        :param t_max: maximum thickness
        :param t_min: minimum thickness
        :param n: data number
        :return: [layer, spec]  layer is [n, mat_lst - 2] thickness data, spec is [n, num] spec data
        """
        mat_lst = self.mat_lst[1:-1]
        n_mat = len(mat_lst)
        layer = torch.zeros(n, n_mat) #n行，n_mat列？不该n_mat行，n列
        for i in range(len(mat_lst)):
            if mat_lst[i] == 'Ag':
                layer[:, i] = torch.randint(15, 30, (n,)) #生成一维向量 n 从15—30的随机数
            else:
                layer[:, i] = torch.randint(t_min, t_max, (n,))
        spec = self.forward(layer)
        return layer, spec

    def loss_function(self, spec):
        device = spec.device.type
        err = torch.mean(self.solar.to(device) * torch.abs(spec - self.target.to(device)))*self.num
        return err

    def loss_function1(self,spec):
        device = spec.device.type
        wl = torch.linspace(400, 800, self.num)
        idx100 = torch.logical_and(wl >= 532, wl < 533)
        idx80 = torch.logical_and(wl >= 533, wl <= 600)
        idx50 = torch.logical_and(wl > 600, wl <= 800)
        idx15 = torch.logical_and(wl < 532, wl > 500)
        idx40 = torch.logical_and(wl >= 400, wl <= 500)
        weight = torch.ones(1, self.num)
        weight[0,idx100] = 700
        weight[0, idx80] = 0.8
        weight[0, idx50] = 0.6
        weight[0, idx15] = 0.8
        weight[0, idx40] = 0.6
        weight = weight.to(device)
        mse = torch.mean(weight * (spec - self.target) ** 2)
        # loss = torch.mean(mse)
        return mse

    def loss_function2(self,spec,input):
        device = spec.device.type
        err1 = torch.mean(self.solar.to(device) * torch.abs(spec - self.target.to(device)))*self.num
        relu = torch.nn.ReLU()
        err2 = torch.mean(1 * relu(torch.abs(input - torch.mean(input)) - 0.5 * (torch.max(input) - torch.min(input))))
        return err1 + err2

    def loss_function3(self,spec,input):
        device = spec.device.type
        wl = torch.linspace(400, 800, self.num)
        idx100 = torch.logical_and(wl >= 532, wl < 533)
        idx80 = torch.logical_and(wl >= 533, wl <= 600)
        idx50 = torch.logical_and(wl > 600, wl <= 800)
        idx15 = torch.logical_and(wl < 532, wl > 500)
        idx40 = torch.logical_and(wl >= 400, wl <= 500)
        weight = torch.ones(1, self.num)
        weight[0, idx100] = 700
        weight[0, idx80] = 0.8
        weight[0, idx50] = 0.6
        weight[0, idx15] = 0.8
        weight[0, idx40] = 0.6
        weight = weight.to(device)
        err1 = torch.mean(weight * (spec - self.target) ** 2)
        relu = torch.nn.ReLU()
        err2 = torch.mean(1 * relu(torch.abs(input - torch.mean(input)) - 0.5 * (torch.max(input) - torch.min(input))))
        return err1 + err2

    def loss_function4(self,spec_var):
        mse = torch.mean(spec_var ** 2)
        return mse


if __name__ == '__main__':
    # mat_lst = ['Y2O3']+['Y2O3', 'HfO2_1', 'Y2O3']*30
    mat_lst = ['MgF2Notch532'] + ['metafilmNotch532']
    mat_lst = ['air'] + mat_lst + ['GlassNotch532']
    #print(mat_lst)
    tm = transfer_matrix(mat_lst=mat_lst, num=401)
    wl = tm.wl
    target = tm.target
    #x0 = [74.22] + [37.11, 68.79, 37.11] * 30
    #x0 = torch.from_numpy(np.array(x0))
    x0_tra = torch.tensor([104.1,52.91])
    x0_opt = torch.tensor([141.3952,27.2280])
    # x0 = torch.tensor([51.6621,  23.1557, 104.0295,  54.7877,  53.0326,  76.4384,  44.7096,
    #      46.8272,  34.9598,  33.2786,  32.6622,  78.1810,  53.7760,  51.0567,
    #      60.5671,  22.7501,  21.0170,  47.5205,  54.4395,  59.1823,  95.2776,
    #      82.0501,  82.0575, 101.8062,  45.4331,  45.5580,  59.0199,  27.3265,
    #      25.8878,  55.7869,  49.8766,  51.1419,  91.6686,  82.3523,  82.1565,
    #     101.7660,  50.2078,  52.2815,  48.5731,  28.4842,  29.8794,  56.5119,
    #      47.8114,  48.3041,  74.1580,  41.7962,  43.6867,  50.8865,  31.7184,
    #      33.1716,  60.5851,  56.1208,  54.9107,  67.0534,  30.7028,  30.7227,
    #      40.5461,  45.3273,  47.5483,  72.8996,  50.7786,  52.4714,  53.7649,
    #      33.1015,  30.7534,  48.5947,  52.5970,  55.5011,  69.4946,  39.1259,
    #      41.0310,  47.6088,  32.5457,  36.7417,  70.3652,  50.8615,  50.1427,
    #      58.7390,  36.9963,  39.4835,  48.4641,  40.8172,  42.8081,  74.2311,
    #      46.5169,  45.2432,  59.7456,  25.2001,  22.8548,  41.6479,  59.3212])  # x0 thickness
    #print(len(x0))
    #x0 = torch.tensor([91.1, 51.6, 190.3, 19.8, 154.1, 109.3])
    #T,_= tm.cal_trans(x0,plot = True)
    T,_,T_0,T_60=tm.cal_trans(x0_tra)
    T_opt,_,T_0_opt,T_60_opt = tm.cal_trans(x0_opt)
    plt.figure(dpi=300)
    plt.plot(wl, target, label='target')
    plt.plot(wl, T_0, label='T_0_tra design')
    plt.plot(wl, T_60, label='T_60_tra design')
    plt.plot(wl, T_0_opt, label='T_0_DL design')
    plt.plot(wl, T_60_opt, label='T_60_DL design')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('transmittance')
    plt.legend()
    plt.show()
    #mse0 = torch.mean((T - tm.target) ** 2) ## the loss between transfer_matrix and target
    loss1 = tm.loss_function1(T)
    loss2 = tm.loss_function2(T,tm.target)
    #print('mse0 = {}'.format(mse0))
    print('loss1 = {}'.format(loss1))
    print(T)
    #print('loss = {}'.format(loss2))
    # x = x.unsqueeze(0).repeat(256, 1)
    #tm.plot_config(x0)
    #T, R, t, r = tm.cal_trans(x0, plot=True)
    #print(torch.mean(tm.target),torch.max(tm.target),torch.min(tm.target))
    # t = time.time()
    # T = tm(x)
    # print(time.time()-t)
    # t = time.time()
    # for i in range(100):
    #     layer, spec = tm.generate_data(10000)
    #     print(time.time() - t, 'seconds')
    #     mse = tm.loss_function(spec)
    #     idx = torch.argmin(mse)
    #     print('min mse = {}'.format(mse[idx]))
    #     tm.plot_config(layer[idx])
    #     plt.plot(tm.wl, T, label='optim design')
    #     plt.plot(tm.wl, spec[0], label='optim design')
    #     plt.plot(tm.wl, tm.target, label='target')
    #     plt.xlabel('wavelength (nm)')
    #     plt.ylabel('trans/refl')
    #     plt.title('pred err {:.5f}, optim err {:.5f}'.format(mse[idx], mse0))
    #     plt.legend()
    #     plt.show()
