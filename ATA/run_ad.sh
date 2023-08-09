RUNS=10
for ((i=0;i<${RUNS};i++));
do
    python main.py --config="pg" --env-config="ad" with env_args.env_name="appleDoor_a" mac="distributed_mac" arch="transformer" rr="ata" local_results_path="results/appleDoor_a/ata" seed=${i}
    python main.py --config="pg" --env-config="ad" with env_args.env_name="appleDoor_b" mac="distributed_mac" arch="transformer" rr="ata" local_results_path="results/appleDoor_b/ata" seed=${i}
done
