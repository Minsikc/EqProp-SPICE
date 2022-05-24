
from sched import scheduler

import pytorch_lightning as pl

from config import cfgBase
from utils.optim import create_optimizer
from torch import Tensor
import torch.nn.functional as F
import torch
import ray
from components.circuits import createCircuit
from components.MyCircuit import MyCircuit
from components.ray_workers import *
from utils.NetlistWriter import SPICEParser
from functools import reduce
from operator import add
import numpy as np
from utils.weightClipper import weightClipper
import torch.nn as nn 
import os

class LitSPICE(pl.LightningModule):
    """models with SPICE backend

    Args:
        pl (_type_): _description_
    """
    
    def __init__(self):
        super().__init__()
        
    pass

class LitEP(pl.LightningModule):
    """models with EP training algorithm

    Args:
        pl (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        
    def forward(self):
        
        pass
    
    def free_phase(self):
        pass
    
    def nudge_phase(self):
        pass

class LightningRay(LitSPICE):

    def __init__(self, dims: list, batch_size: int, num_classes:int, optimizer:str,SPICE_params:dict, mpi_commands:list,optim_kwargs:dict=None, **kwargs):
        super().__init__()

        self.dimensions = dims
        self.n_layers = len(self.dimensions)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.optim_name = optimizer
        self.optim_kwargs = optim_kwargs
        # hparams
        self.SPICE_params = SPICE_params
        self.save_hyperparameters(ignore=["mpi_commands"])
        self._hparams_tolist()

        # weight initialize
        self._weight_init()

        self.Pycircuit = createCircuit(W = self.W, dimensions = self.dimensions, **self.SPICE_params)
        self.circuit = MyCircuit.copyFromCircuit(self.Pycircuit)

        # manual optimization 
        self.automatic_optimization = False
        self.mpi_commands = mpi_commands ###### delete '--allow-run-as-root'
        
        if batch_size == 1:
            self.mpi_commands.pop()
            self.mpi_commands.pop()
            self.mpi_commands.append(str(os.cpu_count() - 2))
            self.mpi_commands.append('-cpu-set')
            self.mpi_commands.append('2-'+str(os.cpu_count()-1))
            print(f'batch size 1 detected. change mpi cmd as {self.mpi_commands}')
        self.clipper = weightClipper(L=self.SPICE_params['L'], U=None)

    def _hparams_tolist(self, keys = ('alpha', 'L', 'U')) -> None:
        """broadcast hyperparameters to list

        Args:
            keys (tuple, optional): _description_. Defaults to ('alpha', 'L', 'U').
        """
        assert self.SPICE_params is not None, 'hparams not set'
        [self.SPICE_params.update({key: [val]*(self.n_layers - 1)}) for (key, val) in self.SPICE_params.items() 
         if key in keys and type(val) is not list]
        assert len(self.SPICE_params['alpha']) == self.n_layers - 1, 'alpha length does not match n_layers'

    def _weight_init(self) -> None:
        assert self.dimensions is not None, 'dimensions not set'
        assert self.SPICE_params is not None, 'hyper_params not set'
        self.W = nn.ModuleList(
        nn.Linear(dim1, dim2, bias=False) #include bias in weight
        for dim1, dim2 in zip(self.dimensions[:-1], self.dimensions[1:])
        )
        assert self.SPICE_params['L'] is not None, 'L not set'
        assert self.SPICE_params['U'] is not None, 'U not set'
        for module, Li, Ui in zip(self.W, self.SPICE_params["L"], self.SPICE_params["U"]):
            module.weight.data = nn.init.uniform_(module.weight, Li, Ui)
    

    def configure_optimizers(self):
        params = []
        [params.append({'params': W.parameters(), 'lr': self.SPICE_params['alpha'][idx]}) for idx, W in enumerate(self.W)]
        optimizer = create_optimizer(params,name=self.optim_name, **self.optim_kwargs)
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        return [optimizer], [lr_scheduler]
        
    
    def training_step(self, batch, batch_idx):
        beta_i = self.SPICE_params['beta'] * [-1, 1][torch.randint(0, 2, (1,))] # random sign
        
        # clone circuit, processed data
        circuit = ray.put(self.circuit)
        batch = ray.put(self._data_preprocess(batch, num_classes=self.num_classes))
        # costFun = ray.put(self.net.costFun)
        ##### parallel processing with ray
            # maybe with loss?
        Vlists = ray.get([ray_train.remote(id, circuit, self.dimensions, batch,
                                                 beta=beta_i, mpi_commands=self.mpi_commands) for id in range(self.batch_size)])
        # merge Vlists
        fdVtuple, ndVtuple, losses = zip(*Vlists)
        
        # fdVtup
        fdVmap = reduce(lambda x, y:map(add, x, y), fdVtuple) 
        self.fdV = np.array(list(fdVmap), dtype = object) / self.batch_size
        ndVmap = reduce(lambda x, y:map(add, x, y), ndVtuple)
        self.ndV = np.array(list(ndVmap), dtype=object) / self.batch_size

     #update everything
        #update G
        # self.net.w_optimize(fdV, ndV, self.optimizers())
        self.zero_grad()
        opt = self.optimizers()
        opt.zero_grad()
        for p, fdv, ndv in zip(self.parameters(), self.fdV, self.ndV):
            p.grad = -(1/beta_i) * torch.from_numpy(ndv**2 - fdv**2).transpose(1,0).float()
            p.grad.contiguous()
        

        #clip weights(G)
        opt.step()
        self.clipper(self.W, 'weight')
        # update Rarray
        SPICEParser.updateWeight(self.circuit, self.W)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", np.array(losses).mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        beta_i = self.SPICE_params['beta'] * [-1, 1][torch.randint(0, 2, (1,))]
        circuit = ray.put(self.circuit)
        (X, Y) = self._data_preprocess(batch, num_classes=self.num_classes)
        X = ray.put(X)
        workers = [ray_predict.remote(id, circuit=circuit, dimensions = self.dimensions, X=X, mpi_commands=self.mpi_commands) for id in range(self.batch_size)]
        outList = ray.get(workers)
        outs = torch.stack(outList, dim=0)
        o1 = outs.clone().detach()
        # calculate output layer grads
        outs.requires_grad = True
        # ypreds = F.softmax(outs, dim=1)
        loss = F.mse_loss(outs, Y, reduction='sum')
        acc = self.accuracy(outs, Y)
        self.log("valid_loss", torch.abs(loss), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("valid_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("confidence", torch.max(o1,dim=1)[0].mean(), on_epoch=True, logger=True)
        
    # def validation_epoch_end(self, output):
    #     [self.tb.add_histogram(name, param, self.current_epoch)for name, param in self.named_parameters()]
        
    def test_step(self, batch, batch_idx):
        
        pass
        
    def predict_step(self, batch, batch_idx):
        circuit = ray.put(self.circuit)
        (X, _) = self._data_preprocess(batch, num_classes=self.num_classes)
        X = ray.put(X)
        workers = [ray_predict.remote(id, circuit=circuit, dimensions = self.dimensions, X=X, mpi_commands=self.mpi_commands) for id in range(self.batch_size)]
        ypredList = ray.get(workers)
        ypreds = torch.stack(ypredList, dim=0)
        return torch.argmax(ypreds, dim=1)

    def accuracy(self, ypreds, labels):
        _, predicted = torch.max(ypreds.data, 1)
        correct = (predicted == torch.argmax(labels,1)).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    @staticmethod
    def _data_preprocess(batch, num_classes=10):
        """process batch data to fit in analog circuit model

        Args:
            batch (_type_): batch dataset
            num_classes (int, optional): number of classification classes. Defaults to 10.

        Returns:
            _type_: processed batch
        """
        X_batch, Y_batch = batch
        X_batch = X_batch.view(X_batch.size(0), -1) #== X_batch.view(-1,X_batch.size(-1)**2)
        X_batch = X_batch.repeat_interleave(2, dim=1)
        X_batch[:,1::2] = -X_batch[:,::2]
        if Y_batch.size(-1) != num_classes:
            Y_batch = F.one_hot(Y_batch, num_classes=num_classes).expand(Y_batch.size(0),-1) 
        return (X_batch, Y_batch)