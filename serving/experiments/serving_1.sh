EXPATH=~/hwlee/Inferencing-CPU-for-network-performance-in-virtualized-environments/serving/experiments
VM=1-test
model_name=random_forest

while IFS=',' read pkt_size bandwidth_tx
do
    python3 $EXPATH/code/main.py $model_name $pkt_size $bandwidth_tx >> cpu_quota.txt
    quota=$(<cpu_quota.txt)
    rm cpu_quota.txt

    cd /sys/fs/cgroup/cpu/machine/qemu-$VM.libvirt-qemu/emulator
    sudo cgset -r cpu.cfs_quota_us=$quota machine/qemu-$VM.libvirt-qemu/emulator
    # sudo cgset -r cpu.cfs_quota_us=100000 machine/qemu-1-test.libvirt-qemu/emulator

    sshpass -p'1' ssh -oStrictHostKeyChecking=no storage@163.152.20.144 "netperf -H 163.152.20.212 -l 120 -- -m 1024" > $EXPATH/data/output_full.txt

    python $EXPATH/code/output.py $model_name $pkt_size $bandwidth_tx $quota
done < $EXPATH/data/input.csv

# graph : output_$model_name.csv 
# [rmsle | actual network throughput | bandwidth_tx(SLO) | pkt_size | quota]
