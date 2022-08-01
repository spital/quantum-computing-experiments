# Quantum Computing Experiments Intro

I recently discovered [IBM Quantum 2022 Updated Development Roadmap](https://www.youtube.com/watch?v=0ka20qanWzI) and was amazed of the recent developments and new possibilities.

I did not waste the time reading Quantum computer programming for dummies [published via Los Alamos National Laboratory LANL news](https://discover.lanl.gov/news/0614-quantum-computer)
because I thought that better was learning by trying ;)

For easy viewing without install and for the test of the new [Github's Math support in markdown](https://github.blog/2022-05-19-math-support-in-markdown/)
I prepared some jupyter notebooks markdown conversions.
Check directories below:
 - [qiskit-finance](./qiskit-finance) - [original repository](https://github.com/Qiskit/qiskit-finance)
 - [qiskit-machine-learning](./qiskit-machine-learning) - [original repository](https://github.com/Qiskit/qiskit-machine-learning)
 - [qiskit-nature](./qiskit-nature) - [original repository](https://github.com/Qiskit/qiskit-nature)
 - [qiskit-optimization](./qiskit-optimization) - [original repository](https://github.com/Qiskit/qiskit-optimization)
 - [qiskit-textbook](./qiskit-textbook) - [original repository](https://github.com/qiskit-community/qiskit-textbook)


## Quckstart

[IBM Quantum computing registration](https://github.com/Qiskit/qiskit-ibmq-provider/blob/master/README.md#configure-your-ibm-quantum-credentials)
is required for some examples, and is recommended for further experiments too. Also you do not need to install anything and just use the cloud.
It is free for basic tasks, simulations, 7 qubits physical hardware too, there is a pay-as-you-go plan for 27 qubit hardware available now (2022/7) also.

## Local Install

If a local install is preferred, using [Ubuntu 20.04 container](https://hub.docker.com/_/ubuntu?tab=tags&page=1&name=20.04) with python worked well.
I had some success with [CUDA enabled cuQuantum Appliance 22.03-cirq from Nvidia](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuquantum-appliance/tags) too,
but sometimes I was not able to start GPU accelerated [IBM's Qiskit Aer backend](https://pypi.org/project/qiskit-aer-gpu/) properly.

Start the container:

``` bash
# For Nvidia GPU:
IMG=nvcr.io/nvidia/cuquantum-appliance:22.03-cirq
docker run -it --network host --name quantum --rm --gpus all -v$(pwd):/shared --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  $IMG bash

# Without Nvidia GPU
IMG=ubuntu:20.04
docker run -it --network host --name quantum --rm --gpus all -v$(pwd):/shared $IMG bash
```

Install requirements:

``` bash
apt update && apt -y install mc zstd sudo iputils-ping net-tools curl wget less iproute2 netcat gnupg git python3-pip

apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

pip install nvidia-pyindex -U

pip install jupyterlab pillow==6.2.2 matplotlib ipywidgets

jupyter nbextension enable --py widgetsnbextension  #       - Validating: OK

pip install numpy==1.22.4 docplex pylatexenc gurobipy cplex sparse torch torchvision pyscf
pip install qiskit-terra qiskit-aer qiskit_ibmq_provider
pip install qiskit-finance qiskit-experiments qiskit-machine-learning qiskit-nature
```

## Run Experiments

Clone any repo of interest e.g. [Qiskit tutorials](https://github.com/Qiskit/qiskit-tutorials), go to `qiskit-tutorials/docs/tutorials`
and start Jupyter Lab via `jupyter lab --allow-root --ip 0.0.0.0`

For some experiments you need to [configure your IBMQ account](https://github.com/Qiskit/qiskit-ibmq-provider/blob/master/README.md#configure-your-ibm-quantum-credentials).

Good luck and May the Force be with you!
