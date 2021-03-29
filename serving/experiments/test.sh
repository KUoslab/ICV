VM=1-test
EXPATH=~/hwlee/Inferencing-CPU-for-network-performance-in-virtualized-environments/serving/experiments
CGPATH=/sys/fs/cgroup/cpu/machine/qemu-$VM.libvirt-qemu/emulator
model_name=random_forest
tmp=0

python3 $EXPATH/code/serving_$model_name.py

while read line
do
    IFS=',' read -a ARR <<< $line
    pkt_size=${ARR[0]}
    bandwidth_tx=${ARR[1]}
    pps_tx=${ARR[2]}
    cpu_usage=${ARR[3]}
    quota=${ARR[4]}
    echo $quota $pkt_size $bandwidth_tx $pps_tx $cpu_usage $quota

    cd $CGPATH
    sudo cgset -r cpu.cfs_quota_us=$quota machine/qemu-$VM.libvirt-qemu/emulator
    
    sshpass -p'1' ssh -oStrictHostKeyChecking=no storage@163.152.20.144 "netperf -H 163.152.20.212 -l 10 -- -m $pkt_size" > $EXPATH/data/output_full.txt
    cd $EXPATH
    python3 $EXPATH/code/output.py $model_name $bandwidth_tx $quota $tmp
    num=$(expr $tmp + 1)
done < $EXPATH/data/input.csv
