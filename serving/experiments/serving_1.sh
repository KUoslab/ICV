PATH=~/Desktop/Inferencing-CPU-for-network-performance-in-virtualized-environments/serving/experiments

while IFS=',' read pkt_size bandwidth_tx
do
    quota=$(python $PATH/code/serving_1.py random_forest $pkt_size $bandwidth_tx)

    cd /sys/fs/cgroup/cpu/machine/qemu-5-test.libvirt-qemu/emulator
    sudo cgset -r cpu.cfs_quota_us=$quota machine/qemu-5-test.libvirt-qemu/emulator

    sshpass -p'1' ssh -oStrictHostKeyChecking=no storage@163.152.20.144 "netperf -H 163.152.20.212 -l 120 -- -m 1024" > $PATH/data/output_full.txt

    python $PATH/code/output.py random_forest $pkt_size $bandwidth_tx $quota
done < $PATH/data/input.csv

while IFS=',' read pkt_size bandwidth_tx
do
    quota=$(python $PATH/code/serving_1.py support_vector $pkt_size $bandwidth_tx)

    cd /sys/fs/cgroup/cpu/machine/qemu-5-test.libvirt-qemu/emulator
    sudo cgset -r cpu.cfs_quota_us=$quota machine/qemu-5-test.libvirt-qemu/emulator

    sshpass -p'1' ssh -oStrictHostKeyChecking=no storage@163.152.20.144 "netperf -H 163.152.20.212 -l 120 -- -m 1024" > $PATH/data/output_full.txt

    python $PATH/code/output.py support_vector $pkt_size $bandwidth_tx $quota
done < $PATH/data/input.csv

# graph : output.csv [actual network throughput | bandwidth_tx | ]
