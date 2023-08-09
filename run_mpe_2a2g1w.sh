RUNS=10
for ((i=0;i<${RUNS};i++));
do
#### Run PPO
    python main.py --env mpe --mpe_fixed_map --mpe_mid_sparse_reward --mpe_use_new_reward --mpe_num_agents 2 --mpe_num_landmarks 2 --mpe_num_walls 1 --algo PPO --frames 20000000 --run ${i} --seed ${i}
#### Run POfD
    python main.py --env mpe --mpe_fixed_map --mpe_mid_sparse_reward --mpe_use_new_reward --mpe_num_agents 2 --mpe_num_landmarks 2 --mpe_num_walls 1 --algo POfD --frames 20000000 --mpe_demonstration 2a2g_1221_wall_noise --pweight 0.1 --pdecay 1 --run ${i} --seed ${i}
done
