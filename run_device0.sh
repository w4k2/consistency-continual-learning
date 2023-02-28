#! /bin/bash

python main.py --method="extended_consistency" --lr=0.001 --seed=1 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-1
python main.py --method="extended_consistency" --lr=0.001 --seed=2 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-1
python main.py --method="extended_consistency" --lr=0.001 --seed=3 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-1
python main.py --method="extended_consistency" --lr=0.001 --seed=4 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-1
python main.py --method="extended_consistency" --lr=0.001 --seed=5 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-1

python main.py --method="extended_consistency" --lr=0.001 --seed=1 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-2
python main.py --method="extended_consistency" --lr=0.001 --seed=2 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-2
python main.py --method="extended_consistency" --lr=0.001 --seed=3 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-2
python main.py --method="extended_consistency" --lr=0.001 --seed=4 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-2
python main.py --method="extended_consistency" --lr=0.001 --seed=5 --n_experiences=20 --regularisation_type="cosine" --n_epochs=50 --device="cuda:0" --use_layer=-2

python main.py --method="replay" --lr=0.001 --seed=1 --n_experiences=20 --n_epochs=50 --device="cuda:0"
python main.py --method="replay" --lr=0.001 --seed=2 --n_experiences=20 --n_epochs=50 --device="cuda:0"
python main.py --method="replay" --lr=0.001 --seed=3 --n_experiences=20 --n_epochs=50 --device="cuda:0"
python main.py --method="replay" --lr=0.001 --seed=4 --n_experiences=20 --n_epochs=50 --device="cuda:0"
python main.py --method="replay" --lr=0.001 --seed=5 --n_experiences=20 --n_epochs=50 --device="cuda:0"
