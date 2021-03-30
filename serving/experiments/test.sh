VM=1-test
EXPATH=~/hwlee/Inferencing-CPU-for-network-performance-in-virtualized-environments/serving/experiments
CGPATH=/sys/fs/cgroup/cpu/machine/qemu-$VM.libvirt-qemu/emulator
model_name=random_forest

python3 $EXPATH/code/serving_$model_name.py

while IFS=',' read pkt_size slo pps cpu quota
do
    cd $CGPATH
    sudo cgset -r cpu.cfs_quota_us='expr "$quota"' machine/qemu-$VM.libvirt-qemu/emulator
    sshpass -p'1' ssh -oStrictHostKeyChecking=no storage@163.152.20.144 "netperf -H 163.152.20.212 -l 1 -- -m $pkt_size" > $EXPATH/data/output_full.txt
    cd $EXPATH
    python3 code/output.py "$model_name" "$slo"
done < $EXPATH/data/input.csv
