RUNS=10
for ((i=0;i<${RUNS};i++));
do
#### Run MAPPO
    python main.py --env appleDoor_a --algo PPO --run ${i} --seed ${i}
    python main.py --env appleDoor_b --algo PPO --run ${i} --seed ${i}
#### Run MAPPO with occupancy measure
    python main.py --env appleDoor_a --algo PPO --use_prior --N 100 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
    python main.py --env appleDoor_b --algo PPO --use_prior --N 100 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
#### Run POfD
    python main.py --env appleDoor_a --algo POfD --pweight 0.02 --pdecay 1 --run ${i} --seed ${i}
    python main.py --env appleDoor_b --algo POfD --pweight 0.02 --pdecay 1 --run ${i} --seed ${i}
done