import random
import numpy as np
import torch
from geneticalgorithm import geneticalgorithm as ga
from forward_model import transfer_matrix as tmm

def f(x):
    arr1, arr2 = np.split(x, 2, axis=0)
    mat_lst = []
    metal = arr1[-1] - len(arr1) + 1
    metal = int(metal)
    # for i in range(len(arr1)):
    #     if(arr1[i] == 0):
    #         mat_lst = mat_lst + ['Ag']
    #     if(arr1[i] == 1):
    #         mat_lst = mat_lst + ['Si']
    for i in range(len(arr1)):
        if(arr1[i] == 0):
            mat_lst = mat_lst + ['SiO2']
        if(arr1[i] == 1):
            mat_lst = mat_lst + ['Si']
        if(arr1[i] == 2):
            mat_lst = mat_lst + ['HfO2']
        if(arr1[i] == 3):
            mat_lst = mat_lst + ['MgF2']
        if(arr1[i] == 4):
            mat_lst = mat_lst + ['TiO2']
        if(arr1[i] == 5):
            mat_lst = mat_lst + ['SiC']
        if(arr1[i] == 6):
            mat_lst = mat_lst + ['ITO']
        if(arr1[i] == 7):
            mat_lst = mat_lst + ['Al2O3']
        if(arr1[i] == 8):
            mat_lst = mat_lst + ['SiN']
        if(arr1[i] == 9):
            mat_lst = mat_lst + ['AZO']
        if(arr1[i] > 9 ):
            insert = arr1[i] - len(mat_lst)
            insert = int(insert)
            mat_lst.insert(insert,'Ag')
    num = arr2[0]
    arr2 = np.delete(arr2,0,axis=0)
    arr2 = np.insert(arr2, metal, num, axis=0)
    mat_lst = ['air']+ mat_lst + ['glass']
    layer = torch.from_numpy(arr2)
    tm = tmm(mat_lst=mat_lst)
    spec,_ = tm.cal_trans(layer)
    #print(tm.loss_function4(spec_var))
    #print(tm.loss_function1(spec))
    loss = tm.loss_function1(spec)
    return loss

if __name__ == '__main__':
    mat_num = 9
    layer_num = 10
    varbound = np.array([[0, 1]]*layer_num+[[layer_num, 2*layer_num]]+[[15, 30]]+[[15, 300]]*layer_num)
    vartype = np.array([['int']]*10+[['int']]+[['int']]+[['int']]*10)
    # varbound = np.array([[0, 1]]*layer_num+[[15, 300]]*layer_num)
    # vartype = np.array([['int']]*10+[['int']]*10)
    algorithm_param = {'max_num_iteration':100,
                       'population_size': 100,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': None,
                       'function_timeout':30}
    model =ga(function = f,dimension = 20,variable_type_mixed = vartype,variable_boundaries=varbound,algorithm_parameters=algorithm_param)
    model.run()
    model.report()
    model.param()
    model.output_dict()
    # t_min = 0
    # t_max = 400
    # str = 'SiO2	Al2O3	HfO2	Al2O3	Al2O3	MgF2	TiO2	MgF2	MgF2	TiO2	SiC	ITO	SiO2	TiO2	MgF2	SiC	Al2O3	SiO2	HfO2	TiO2	MgF2	SiC	MgF2	Al2O3	MgF2'
    # str_lst = str.split()
    # random.shuffle(str_lst)
    # mat_lst = ['air'] + str_lst + ['air']
    # mat_lst_1 = mat_lst[1:-1]
    # n_mat = len(mat_lst)
    # print(n_mat)
    # layer = torch.zeros(1,n_mat-2)  # n行，n_mat列？不该n_mat行，n列
    # print(len(layer))
    # for i in range(len(mat_lst_1)):
    #     if mat_lst_1[i] == 'Ag':
    #         layer[:, i] = torch.randint(15, 30,(1,))  # 生成一维向量 n 从15—30的随机数
    #     else:
    #         layer[:, i] = torch.randint(t_min, t_max,(1,))
    # print(len(layer),layer)
    # tm = tmm(mat_lst=mat_lst)
    # spec, _, _, _ = tm.cal_trans_batch(layer)




