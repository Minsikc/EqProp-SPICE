from components.Litnet import LightningRay
from datamodules import load_dataset

from config import *
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
from glob import glob

def main(ckpt:bool=False,path:str='miniMNIST16-original', version:str=None, config=None, project_name:str=None,):
  cfg = config if issubclass(type(config) ,cfgBase) else miniMNIST16cfg()
  
  cfg.SPICE_params['Diode']['Path'] = './ex/libraries/diode/switching/1N4148.lib'
  cfg.batch_size = os.cpu_count() - 2
  print(cfg.to_dict())

  n = cfg.frac
  dataset = load_dataset('mini_MNIST', selected_classes=[0,1,2,3,4], samples_per_cls=65*n, standard_deviation = cfg.std_dev) #16-by-16 

  num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else cfg.num_classes #mini_MNIST
  train_dataset, val_dataset = random_split(dataset, [60*n*5,5*n*5])

  train_loader = DataLoader(train_dataset, num_workers=1, persistent_workers=False,
   batch_size = cfg.batch_size, drop_last = True, shuffle=True)
  val_loader = DataLoader(val_dataset, num_workers=1, persistent_workers=False,
   batch_size = cfg.batch_size, drop_last = True, shuffle=False)
  
  if ckpt:
    ckpt_path = os.path.join("tb_logs", path, version, 'checkpoints')
    ckpt_files = glob(os.path.join(ckpt_path, '*.ckpt'))
    model = LightningRay.load_from_checkpoint(ckpt_files[-1], mpi_commands=cfg.mpi_commands)
  else:
    model = LightningRay(**cfg.to_dict())
  # seed set to 42 for reproducibility
  pl.seed_everything(42, workers=True)
  # logger 
  tb_logger = TensorBoardLogger(save_dir="tb_logs", name=path)
  
  loggers = [tb_logger]
  
  trainer = pl.Trainer(logger = loggers, max_epochs= cfg.num_epochs)
  trainer.fit(model= model, train_dataloaders= train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
  ## for xyce
  cfg = miniMNIST16cfg({'frac':1, 'num_epochs':10, 'upper_frac':2.8, 'optimizer':'adam', 'std_dev':1,'SPICE_params/A':4,
                        'SPICE_params/Diode/Rectifier':'BidRectifier','SPICE_params/L':1e-5})
  main(path='miniMNIST16-W', project_name='Eqprop-spice', config=cfg)
