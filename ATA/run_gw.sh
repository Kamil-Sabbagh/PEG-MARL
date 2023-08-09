RUNS=10
for ((i=0;i<${RUNS};i++));
do
#    python main.py --config="pg" --env-config="gw" with env_args.env_name="centerSquare6x6_2a" mac="distributed_mac" arch="transformer" rr="ata" local_results_path="results/centerSquare6x6_2a/ata" seed=${i}
    python main.py --config="pg" --env-config="gw" with env_args.env_name="centerSquare6x6_3a" mac="distributed_mac" arch="transformer" rr="ata" local_results_path="results/centerSquare6x6_3a/ata" seed=${i}
#    python main.py --config="pg" --env-config="gw" with env_args.env_name="centerSquare6x6_4a" mac="distributed_mac" arch="transformer" rr="ata" local_results_path="results/centerSquare6x6_4a/ata" seed=${i}
done
