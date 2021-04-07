num=1
while IFS=',' read real_quota pkt_size bandwidth pps cpu_usage expected_quota
do
	sudo cgset -r cpu.cfs_quota_us=$expected_quota machine/qemu-19-test1.libvirt-qemu/emulator
	cat /sys/fs/cgroup/cpu/machine/qemu-19-test1.libvirt-qemu/emulator/cpu.cfs_quota_us
	netperf -H 192.168.122.149 -l 120 -- -m $pkt_size > data/output/output_$num.txt
	num=$(($num+1))
done < data/test.csv
