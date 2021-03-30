#!/bin/sh
if [ $# -ne 1 ];then
        echo "usage: sh pps.sh <interface>"
        echo "ex) sh pps.sh eth0"
        exit 1;
fi
IFACE=$1
TX1=`ifconfig $IFACE | grep 'TX packets' | awk '{print $2}' | awk -F ':' '{print $2}'`
RX1=`ifconfig $IFACE | grep 'RX packets' | awk '{print $2}' | awk -F ':' '{print $2}'`
echo -e "DATETIME\t\tTX\tRX"
while [ 1 ]
do
        sleep 1
        TX2=`ifconfig $IFACE | grep 'TX packets' | awk '{print $2}' | awk -F ':' '{print $2}'`
        RX2=`ifconfig $IFACE | grep 'RX packets' | awk '{print $2}' | awk -F ':' '{print $2}'`
        TX_DIFF=$(($TX2-$TX1))
        RX_DIFF=$(($RX2-$RX1))
        NOW=`date +%Y-%m-%d\ %H:%M:%S`
        echo -e "$NOW\t$TX_DIFF\t$RX_DIFF"
        TX1=$TX2
        RX1=$RX2
done