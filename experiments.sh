#!/bin/bash

counter=1
while [ $counter -le 5]
do
    python3 run.py --log_file=Logs/supervised.txt --query_strategy=supervised
    python3 run.py --log_file=Logs/random.txt --query_strategy=random
    python3 run.py --log_file=Logs/least_confident.txt --query_strategy=least_confident
    python3 run.py --log_file=Logs/least_confident_dropout.txt --query_strategy=least_confident_dropout
    python3 run.py --log_file=Logs/margin.txt --query_strategy=margin
    python3 run.py --log_file=Logs/margin_dropout.txt --query_strategy=margin_dropout
    python3 run.py --log_file=Logs/entropy.txt --query_strategy=entropy
    python3 run.py --log_file=Logs/entropy_dropout.txt --query_strategy=entropy_dropout
    python3 run.py --log_file=Logs/bald.txt --query_strategy=bald
    python3 run.py --log_file=Logs/kmeans.txt --query_strategy=kmeans
    python3 run.py --log_file=Logs/kcentre.txt --query_strategy=kcentre
    python3 run.py --log_file=Logs/core_set.txt --query_strategy=core_set
    python3 run.py --log_file=Logs/deep_fool.txt --query_strategy=deep_fool
    ((counter++))
done
