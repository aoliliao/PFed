python main.py -nc 5 -jr 1 -nb 10 -data digits -m cnn -algo FedBN

python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -ls 10 -algo FedBN
python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -algo FedAvg
python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn --seed 1  -com seed1 -algo Fed
python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -algo FedBABU
python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -ls 10 -algo FedRe
python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -com 1:0 -algo Fed

python main.py -nc 4 -jr 1 -nb 10 -data office -m cnn -com 0.5* -algo FedAvg

for pickle cifar100
python main.py -nc 5 -jr 1 -nb 20  -lr 0.001 -lbs 40 -data Cifar -m cnn -algo Fed
python main.py -nc 5 -jr 1 -nb 20  -lr 0.001 -lbs 40 -data Cifar -m cnn -algo FedBN
python main.py -nc 5 -jr 1 -nb 20  -lr 0.001 -lbs 40 -data Cifar -m cnn -algo FedAvg

for cifar20
python main.py -nc 50 -jr 1 -nb 20  -lr 0.01 -lbs 40 -data Cifar -m cnn -algo Fed
python main.py -nc 50 -jr 1 -nb 20  -lr 0.01 -lbs 40 -data Cifar -m cnn -algo FedBN
python main.py -nc 50 -jr 1 -nb 20  -lr 0.01 -lbs 40 -data Cifar -m cnn -algo FedAvg
python main.py -nc 50 -jr 1 -nb 20  -lr 0.01 -lbs 40 -data Cifar -m cnn -algo FedBABU


python main.py -nc 10 -jr 1 -nb 20  -lr 0.01 -lbs 40 -data Cifar -m cnn -algo Fed


for digit5
python main.py -nc 5 -jr 1 -nb 10 -data digits -m cnn -algo FedBN
python main.py -nc 5 -jr 1 -nb 10 -data digits -m cnn -algo Fed
python main.py -nc 5 -jr 1 -nb 10 -data digits -m cnn -algo FedAvg

python main.py -nc 6 -jr 1 -nb 10 -data domainnet -m cnn -algo FedAvg
python main.py -nc 6 -jr 1 -nb 10 -data domainnet -m cnn  -com 1:0 -algo Fed

python main.py -nc 6 -jr 1 -nb 10 -data domainnet -m cnn -did 1 -com did1  -gr 300 -algo FedAvg