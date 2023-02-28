#! /bin/bash

python main.py --method="extended_consistency" --lr=0.001 --seed=1 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=5
python main.py --method="extended_consistency" --lr=0.001 --seed=2 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=5
python main.py --method="extended_consistency" --lr=0.001 --seed=3 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=5
python main.py --method="extended_consistency" --lr=0.001 --seed=4 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=5
python main.py --method="extended_consistency" --lr=0.001 --seed=5 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=5

python main.py --method="extended_consistency" --lr=0.001 --seed=1 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=20
python main.py --method="extended_consistency" --lr=0.001 --seed=2 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=20
python main.py --method="extended_consistency" --lr=0.001 --seed=3 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=20
python main.py --method="extended_consistency" --lr=0.001 --seed=4 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=20
python main.py --method="extended_consistency" --lr=0.001 --seed=5 --n_experiences=20 --regularisation_type="KD" --n_epochs=50 --device="cuda:1" --T=20

python main.py --method="extended_consistency" --lr=0.001 --seed=1 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-1
python main.py --method="extended_consistency" --lr=0.001 --seed=2 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-1
python main.py --method="extended_consistency" --lr=0.001 --seed=3 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-1
python main.py --method="extended_consistency" --lr=0.001 --seed=4 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-1
python main.py --method="extended_consistency" --lr=0.001 --seed=5 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-1

python main.py --method="extended_consistency" --lr=0.001 --seed=1 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-2
python main.py --method="extended_consistency" --lr=0.001 --seed=2 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-2
python main.py --method="extended_consistency" --lr=0.001 --seed=3 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-2
python main.py --method="extended_consistency" --lr=0.001 --seed=4 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-2
python main.py --method="extended_consistency" --lr=0.001 --seed=5 --n_experiences=20 --regularisation_type="Linf" --n_epochs=50 --device="cuda:1" --use_layer=-2