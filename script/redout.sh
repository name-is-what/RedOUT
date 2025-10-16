#!/bin/bash

python main.py -exp_type oodd -DS_pair ogbg-molfreesolv+ogbg-moltoxcast -batch_size_test 128

python main.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY

python main.py -exp_type oodd -DS_pair PTC_MR+MUTAG

python main.py -exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -batch_size_test 128

python main.py -exp_type oodd -DS_pair ogbg-moltox21+ogbg-molsider -batch_size_test 128

python main.py -exp_type oodd -DS_pair AIDS+DHFR -batch_size_test 128

python main.py -exp_type oodd -DS_pair ogbg-molbbbp+ogbg-molbace -batch_size_test 128

python main.py -exp_type oodd -DS_pair BZR+COX2

python main.py -exp_type oodd -DS_pair ogbg-molclintox+ogbg-mollipo -batch_size_test 128

python main.py -exp_type oodd -DS_pair ENZYMES+PROTEINS