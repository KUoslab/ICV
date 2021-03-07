# pip install --upgrade pip
# pip install openpyxl
# pip install pandas
python code/input.py

while read quota
do
    sudo "sed -i \"1s/.*/$quota/g\" /sys/fs/cgroup/cpu/machine/qemu-5-test.libvirt-qemu/emulator/cpu.cfs_quota_us"

    # connect on netserver
    ssh-keygen -f "/home/test/.ssh/known_hosts" -R "163.152.20.144"
    # sudo apt install sshpass 
    sshpass -p'1' ssh -oStrictHostKeyChecking=no storage@163.152.20.144 "netperf -H 163.152.20.212 -l 120 -- -m 1024" > data/output_full.txt

    python code/output.py $quota
    sleep 1s
done < data/input.csv

# 그래프 그리기 : output.csv [cpu quota / network throughput]