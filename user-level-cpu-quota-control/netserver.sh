# pip install --upgrade pip
# pip install openpyxl
# pip install pandas
# generate data: input.csv [cpu quota]
PATH=~/Desktop/Inferencing-CPU-for-network-performance-in-virtualized-environments/user-level-cpu-quota-control
python $PATH/code/input.py

while read quota
do
    # sudo apt install cgroup-tools
    # quota > cpu.cfs_quota_us
    echo $quota
    cd /sys/fs/cgroup/cpu/machine/qemu-5-test.libvirt-qemu/emulator
    sudo cgset -r cpu.cfs_quota_us=$quota machine/qemu-5-test.libvirt-qemu/emulator
    #sed -i "1s/.*/$quota/g" /sys/fs/cgroup/cpu/machine/qemu-5-test.libvirt-qemu/emulator/cpu.cfs_quota_us

    # sudo ssh-keygen -f "/root/.ssh/known_hosts" -R "163.152.20.144"
    # sudo apt install sshpass 
    # netperf command result > output_full.txt
    sshpass -p'1' ssh -oStrictHostKeyChecking=no storage@163.152.20.144 "netperf -H 163.152.20.212 -l 120 -- -m 1024" > $PATH/data/output_full.txt

    # generate data: output.csv [cpu quota, network throughput]
    python $PATH/code/output.py $quota
    sleep 1s
done < $PATH/data/input.csv

# graph : output.csv [cpu quota / network throughput]
