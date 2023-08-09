RUNS=10
for ((i=0;i<${RUNS};i++));
do
#### Run MAPPO
    python main.py --env centerSquare6x6_2a --algo PPO --frames 5000000 --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_3a --algo PPO --frames 5000000 --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_4a --algo PPO --frames 5000000 --run ${i} --seed ${i}
#### Run MAPPO with occupancy measure
    python main.py --env centerSquare6x6_2a --algo PPO --use_prior --N 100 --pweight 1 --pdecay 0.995 --add_noise --frames 5000000 --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_3a --algo PPO --use_prior --N 100 --pweight 1 --pdecay 0.995 --add_noise --frames 5000000 --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_4a --algo PPO --use_prior --N 100 --pweight 1 --pdecay 0.995 --add_noise --frames 5000000 --run ${i} --seed ${i}
#### Run POfD
    python main.py --env centerSquare6x6_2a --algo POfD --pweight 0.02 --pdecay 1 --frames 5000000 --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_3a --algo POfD --pweight 0.02 --pdecay 1 --frames 5000000 --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_4a --algo POfD --pweight 0.02 --pdecay 1 --frames 5000000 --run ${i} --seed ${i}
done
