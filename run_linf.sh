#! /bin/bash

python main.py --method="extended_consistency" --lr=0.003 --seed=1 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50
python main.py --method="extended_consistency" --lr=0.003 --seed=2 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50
python main.py --method="extended_consistency" --lr=0.003 --seed=3 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50
python main.py --method="extended_consistency" --lr=0.003 --seed=4 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50
python main.py --method="extended_consistency" --lr=0.003 --seed=5 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50