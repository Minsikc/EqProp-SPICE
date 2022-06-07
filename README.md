# EqProp-SPICE
Unofficial Implementation of Equilibrium-Propagation on analog circuit with SPICE simulator and Pytorch Lightning as introduced in [Training End-to-End Analog Neural Networks with Equilibrium Propagation](https://arxiv.org/abs/2006.01981)

# Installation
1. Build SPICE simulator (Xyce) from sandia.gov
```shell
setup_xyce.sh
```
2. install all requirements
```shell
pip -r requirements.txt
```
3. download diode libaries from pyspice & unzip to ./ex subdirectory
```shell
pyspice-post-installation --download-example
```
4. change