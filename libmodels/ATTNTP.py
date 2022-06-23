import torch
import tqdm
from libmodels.multiheaded_attention import MultiHeadedAttention as MHA
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class ATTN_TP(nn.Module):
    def __init__(self, struct_dict: dict):
        super().__init__()
        for key in struct_dict:
            self.__setattr__(key, struct_dict[key])
            if key == 'loss_function':
                self.__setattr__(key, struct_dict[key]())

        latlon_bias, alt_bias = False, False
        if hasattr(self,'latlon_wb'):
            if 'b_k' in self.latlon_wb:
                if self.latlon_wb['b_k'] != 0: latlon_bias = True
            if 'b_v' in self.latlon_wb:
                if self.latlon_wb['b_k'] != 0: latlon_bias = True
        if hasattr(self, 'alt_wb'):
            if 'b_k' in self.alt_wb:
                if self.alt_wb['b_k'] != 0: alt_bias = True
            if 'b_v' in self.alt_wb:
                if self.alt_wb['b_k'] != 0: alt_bias = True

        self.latlon = torch.nn.MultiheadAttention(2,1,add_bias_kv=True,kdim=400,vdim=800)
        self.alt = torch.nn.MultiheadAttention(2,1,add_bias_kv=True,kdim=800,vdim=400)

        if hasattr(self, 'latlon_wb'):
            qshape, kshape, vshape = self.latlon.q_proj_weight.shape, self.latlon.k_proj_weight.shape, self.latlon.v_proj_weight.shape
            self.latlon.q_proj_weight.data = torch.diag(torch.full((qshape[0],), self.latlon_wb['w_q']))
            self.latlon.k_proj_weight.data = torch.diag(torch.full((int(kshape[1]**.5),), self.latlon_wb['w_k'])).repeat(2,1,1).view(2,-1)
            self.latlon.v_proj_weight.data = torch.stack((torch.cat((torch.diag(torch.full((20,),
                                               self.latlon_wb['w_v'])), torch.zeros(20,20)), 1),
                                               torch.cat((torch.zeros(20,20), torch.diag(torch.full((20,),
                                               self.latlon_wb['w_v']))), 1)),0).view(2,-1)
            # self.latlon.bias_q.data.fill_(self.latlon_wb['b_q'])
            self.latlon.bias_k.data.fill_(self.latlon_wb['b_k'])
            self.latlon.bias_v.data.fill_(self.latlon_wb['b_v'])
        if hasattr(self, 'alt_wb'):
            qshape, kshape, vshape = self.alt.q_proj_weight.shape, self.alt.k_proj_weight.shape, self.alt.v_proj_weight.shape
            self.alt.q_proj_weight.data = torch.diag(torch.full((qshape[0],), self.alt_wb['w_q']))
            self.alt.k_proj_weight.data = torch.stack((torch.cat((torch.diag(torch.full((20,),
                                               self.alt_wb['w_k'])), torch.zeros(20,20)), 1), torch.cat((torch.zeros(20,20),
                                               torch.diag(torch.full((20,), self.alt_wb['w_k']))), 1)),0).view(2,-1)
            self.alt.v_proj_weight.data = torch.diag(torch.full((int(vshape[1]**.5),), self.alt_wb['w_v'])).repeat(2,1,1).view(2,-1)
            # self.latlon.bias_q.data.fill_(self.alt_wb['b_q'])
            self.alt.bias_k.data.fill_(self.alt_wb['b_k'])
            self.alt.bias_v.data.fill_(self.alt_wb['b_v'])

        if 'cuda' in self.device:
            self.cuda(self.device.split(':')[1])
            self.device = '{}:{}'.format(self.latlon.bias_k.device.type, self.latlon.bias_k.device.index)

        if hasattr(self, 'optim'):
            if struct_dict['optim'] == 'sgd':
                # lr, mom, dec, nest= 0.001, 0.0, struct_dict['weight_reg'], False
                lr, mom, dec, nest = 0.01, 0.7, struct_dict['weight_reg'], True
                if 'forcelr' in struct_dict: lr = struct_dict['forcelr']
                if 'forcemom' in struct_dict: mom = struct_dict['forcemom']
                self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=dec, momentum=mom, nesterov=nest)
            elif struct_dict['optim'] == 'sgd+momentum':
                lr, mom, dec, nest = 0.001, 0.0, struct_dict['weight_reg'], False
                if 'forcelr' in struct_dict: lr = struct_dict['forcelr']
                if 'forcemom' in struct_dict: mom = struct_dict['forcemom']
                self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-5, momentum=0.5, nesterov=False,
                                                weight_decay=dec)
            elif struct_dict['optim'] == 'sgd+nesterov':
                lr, mom, dec, nest = 0.001, 0.0, struct_dict['weight_reg'], False
                if 'forcelr' in struct_dict: lr = struct_dict['forcelr']
                if 'forcemom' in struct_dict: mom = struct_dict['forcemom']
                self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=mom, nesterov=True, weight_decay=dec)
            elif struct_dict['optim'] == 'rmsprop':
                lr = 0.001
                if 'forcelr' in struct_dict: lr = struct_dict['forcelr']
                self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=struct_dict['weight_reg'])
            elif struct_dict['optim'] == 'adadelta':
                lr = 1.
                if 'forcelr' in struct_dict: lr = struct_dict['forcelr']
                self.optimizer = torch.optim.Adadelta(self.parameters(), lr=lr, weight_decay=struct_dict['weight_reg'])
            elif struct_dict['optim'] == 'adagrad':
                lr = 0.001
                if 'forcelr' in struct_dict: lr = struct_dict['forcelr']
                self.optimizer = torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=struct_dict['weight_reg'])
            else:
                lr = 0.001
                if 'forcelr' in struct_dict: lr = struct_dict['forcelr']
                self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=struct_dict['weight_reg'])
            if 'forcestep' in struct_dict and 'forcegamma' in struct_dict:
                self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=struct_dict['forcestep'],
                                                            gamma=struct_dict['forcegamma'])
        self.struct_dict = vars(self)



    def forward(self, et_data, et_coord, x_fp):
        etshape = et_data.shape
        et_data = et_data.view(etshape[0],etshape[1],400)
        et_coord = torch.cat((et_coord[:,:,:,:,0].view(etshape[0],etshape[1],400), et_coord[:,:,:,:,1].view(etshape[0],etshape[1],400)),2)
        latlon, latlon_weights = self.latlon(x_fp[:,:,:2], et_data, et_coord)
        alt_V = torch.tensor(F.relu(et_data - x_fp[:,:,2].reshape(etshape[0],etshape[1],1)), device=self.device)
        alt, alt_weights = self.alt(latlon, et_coord, alt_V)
        pred = torch.cat((latlon, alt.mean(2).view(etshape[0],etshape[1],1)),2)
        return pred


    #def init_w_b_const(self):



    def update_dict(self):
        del self.struct_dict
        exclude = ['training', '_parameters', '_buffers', '_non_persistent_buffers_set', '_backward_hooks',
                   '_is_full_backward_hook', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks',
                   '_load_state_dict_pre_hooks', '_modules']
        self.struct_dict = {x: self.__getattribute__(str(x)) for x in vars(self) if not x in exclude}

    def save_model(self, override_path: str = None, override: bool = False):
        model_name = self.model_name()
        if override_path == None:
            container = 'Models/{}/{}/'.format('&'.join(self.features),model_name)
            model_path = '{}/{}'.format(container, model_name)
            while os.path.isfile(model_path) and not override:
                choice = input("Model Exists:\n1: Replace\n2: New Model\n")
                if choice == '1':
                    break
                elif choice == '2':
                    name = input("Enter model name\n")
                    model_path = '{}/{}'.format(container,name)
            container = '/'.join(model_path.split('/')[:-1])
        else:
            container = override_path
            model_path = '{}/{}'.format(container, model_name)
        if not os.path.isdir(container):
            os.makedirs(container)
        torch.save({'struct_dict': self.struct_dict, 'state_dict': self.state_dict(),
                    'opt_dict': self.optimizer.state_dict(), 'epochs_trained': self.epochs_trained,
                    'batch_size': self.batch_size}, model_path)
    def model_name(self):
        if hasattr(self,'name'): return self.name
        else: return 'CrossATTN-2lay'