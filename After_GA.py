import torch
from forward_model import transfer_matrix as tmm
import numpy as np

if __name__ == '__main__':
    materials = [  6,   0,   8,   2,   3,   2,   5,   2,   2,   8,  16]
    mat_lst =[]
    for i in range(len(materials)):
        if(materials[i] == 0):
            mat_lst = mat_lst + ['SiO2']
        if(materials[i] == 1):
            mat_lst = mat_lst + ['Al2O3']
        if(materials[i] == 2):
            mat_lst = mat_lst + ['HfO2']
        if(materials[i] == 3):
            mat_lst = mat_lst + ['MgF2']
        if(materials[i] == 4):
            mat_lst = mat_lst + ['TiO2']
        if(materials[i] == 5):
            mat_lst = mat_lst + ['SiC']
        if(materials[i] == 6):
            mat_lst = mat_lst + ['ITO']
        if (materials[i] == 7):
            mat_lst = mat_lst + ['Si']
        if (materials[i] == 8):
            mat_lst = mat_lst + ['SiN']
        if (materials[i] == 9):
            mat_lst = mat_lst + ['AZO']
        if (materials[i] > 9):
            insert = materials[i] - len(mat_lst)
            insert = int(insert)
            mat_lst.insert(insert, 'Ag')
    mat_lst = ['air'] + mat_lst +['glass']
    x0 =torch.tensor([15,  56, 134,
 130, 187, 151,  70, 198, 107, 193,  82])
    metal = materials[-1] - len(materials) + 1
    metal = int(metal)
    x0 = x0.numpy()
    num = x0[0]
    x0 = np.delete(x0,0,axis=0)
    x0 = np.insert(x0, metal, num, axis=0)
    x0 = torch.from_numpy(x0)
    tm = tmm(mat_lst = mat_lst)
    spec, _, _, _ = tm.cal_trans(x0)
    loss = tm.loss_function1(spec)
    print(mat_lst)
    print(x0)
    print('loss is {}'.format(loss))
    tm.plot_config(x0)
    T, R, t, r = tm.cal_trans(x0, plot=True)

