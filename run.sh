#!/bin/bash

python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -algo Fed5 -did 1 -com didi_LR0.001 -lr 0.001
python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -algo FedAvg -did 1 -com didi_LR0.001 -lr 0.001


#python main.py -nc 6 -jr 1 -nb 10 -data domainnet -m cnn -algo FedNet11 -com did1_catgate -did 1
