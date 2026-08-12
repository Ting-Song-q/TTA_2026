#!/bin/bash

# 自己的ip
TARGET_IP="192.168.31.89"

# 查找TIME-WAIT状态的连接并终止
echo "Terminating TIME-WAIT connections for IP $TARGET_IP"
for port in $(ss -tan state time-wait '( dport = :* or sport = :* )' | grep $TARGET_IP | awk '{print $4}' | cut -d: -f2 | sort -u); do
    fuser -k -n tcp $port
done

# 查找LISTEN状态的连接并终止
echo "Terminating LISTEN connections for IP $TARGET_IP"
for port in $(ss -tan state listen '( dport = :* or sport = :* )' | grep $TARGET_IP | awk '{print $4}' | cut -d: -f2 | sort -u); do
    fuser -k -n tcp $port
done

echo "Terminations complete."
TARGET_IP="192.168.31.110"

# 查找TIME-WAIT状态的连接并终止
echo "Terminating TIME-WAIT connections for IP $TARGET_IP"
for port in $(ss -tan state time-wait '( dport = :* or sport = :* )' | grep $TARGET_IP | awk '{print $4}' | cut -d: -f2 | sort -u); do
    fuser -k -n tcp $port
done

# 查找LISTEN状态的连接并终止
echo "Terminating LISTEN connections for IP $TARGET_IP"
for port in $(ss -tan state listen '( dport = :* or sport = :* )' | grep $TARGET_IP | awk '{print $4}' | cut -d: -f2 | sort -u); do
    fuser -k -n tcp $port
done

echo "Terminations complete."