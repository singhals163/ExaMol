#!/bin/bash

sudo apt-get update
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
bash Anaconda3-2025.06-1-Linux-x86_64.sh    
rm Anaconda3-2025.06-1-Linux-x86_64.sh
export export PATH=$HOME/anaconda3/bin:$PATH

conda env create --file envs/environment-cpu.yml --yes
conda init bash
conda activate examol
