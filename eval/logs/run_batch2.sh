#!/bin/bash
cd /Users/albert/Projects/philo

nohup python3 -m eval.run --mode iterate \
    --tasks private_detectives_and_investigators__57b2cdf2 \
    --rounds 3 --repeats 2 --temperature 0.7 \
    > /Users/albert/Projects/philo/eval/logs/detective_57b2cdf2.log 2>&1 &

nohup python3 -m eval.run --mode iterate \
    --tasks private_detectives_and_investigators__a46d5cd2 \
    --rounds 3 --repeats 2 --temperature 0.7 \
    > /Users/albert/Projects/philo/eval/logs/detective_a46d5cd2.log 2>&1 &

nohup python3 -m eval.run --mode iterate \
    --tasks producers_and_directors__e4f664ea \
    --rounds 3 --repeats 2 --temperature 0.7 \
    > /Users/albert/Projects/philo/eval/logs/producer_e4f664ea.log 2>&1 &

echo "All 3 batch2 tasks launched with nohup"
wait
echo "=== Batch 2 completed ==="
