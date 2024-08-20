#!/bin/bash
ACCELERATE_CONFIG=$HOME/.cache/huggingface/accelerate/default_config.yaml
output=$(python -c "import idr_torch; print(f'{idr_torch.master_addr},{idr_torch.master_port}')")
MASTER_ADDRESS=$(echo $output | cut -d, -f1)
MASTER_PORT=$(echo $output | cut -d, -f2)
sed -i "s/^main_process_ip:.*/main_process_ip: $MASTER_ADDRESS/" $ACCELERATE_CONFIG
sed -i "s/^main_process_port:.*/main_process_port: $MASTER_PORT/" $ACCELERATE_CONFIG
