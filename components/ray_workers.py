import ray
from components.MyCircuit import MyCircuit
from components.xyce import XyceSim
from utils.NetlistWriter import SPICEParser
import torch.nn.functional as F
import torch
import numpy as np

# SPICE wrapper for xyce_parallel
@ray.remote
def ray_train(id:int, circuit:MyCircuit, dimensions:list , batch, beta, mpi_commands=None, normalize_per_input:bool=False): # id:int):
    """_summary_

    Args:
        circuit (Circuit): _description_
        dimensions (list): _description_
        x (_type_): _description_
        y (_type_): _description_
        mpi_commands (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # setup
    X, Y = batch
    x = X[id,:]
    y = Y[id,:]
    # normalize input with torch.var_mean()
    if normalize_per_input:
        (std, mean) = torch.std_mean(x)
        x = (x - mean) / std

    if mpi_commands[-1] == '-cpu-set':
        mpi_commands.append(str(id + 2))
    # free phase
    SPICEParser.clampLayer(circuit, x)
    #analysis
    xyce = XyceSim(mpi_command = mpi_commands)
    raw_file = xyce(spice_input = circuit)
    voltages = SPICEParser.fastRawfileParser(raw_file, nodenames = circuit.nodes, dimensions = dimensions)
    free_Vdrops, ypred = SPICEParser.w_get_Vdrop_w_ypred(voltages)

    # calculate output layer grads
    
    ypred = ypred.expand(1,-1)
    ypred.requires_grad = True
    loss = F.mse_loss(ypred, y.expand(1,-1).double(), reduction='sum')
    # loss = costFun.compute_energy(ypred)
    loss.backward()
    ygrad = ypred.grad*beta
    #nudged phase
    SPICEParser.releaseLayer(circuit, ygrad)
    # analysis 2
    raw_file2 = xyce(spice_input = circuit)
    voltages2 = SPICEParser.fastRawfileParser(raw_file2, nodenames = circuit.nodes, dimensions = dimensions)
    nudged_Vdrops, _ = SPICEParser.w_get_Vdrop_w_ypred(voltages2)

    return (free_Vdrops, nudged_Vdrops, np.abs(loss.detach().numpy()))

@ray.remote
def ray_predict(id, circuit:MyCircuit, dimensions:list , X, mpi_commands=None):
    """_summary_

    Args:
        circuit (Circuit): _description_
        dimensions (list): _description_
        x (Tensor): input tensor
        mpi_commands (_type_, optional): _description_. Defaults to None.

    Returns:
        Tensor(1,num_classes): prediction per classes
    """
    x = X[id,:]
    # input_dimension = dimensions[0]
    # output_dimension = dimensions[-1]
    if mpi_commands[-1] == '-cpu-set':
        mpi_commands.append(str(id + 2))
    
    # free phase
    SPICEParser.clampLayer(circuit, x)
    #analysis
    xyce = XyceSim(mpi_command = mpi_commands)
    raw_file = xyce(spice_input = circuit)
    (_, Vout) = SPICEParser.fastRawfileParser(raw_file, nodenames = circuit.nodes, dimensions = dimensions)
    out = Vout[-1][::2]-Vout[-1][1::2]
    out = torch.from_numpy(out)
    return out