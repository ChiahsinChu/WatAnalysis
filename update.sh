ssh -p 6666 -i ~/.ssh/id_rsa jxzhu@172.27.127.191 "rm -r /data/home/jxzhu/.conda/envs/work_env/lib/python3.9/site-packages/zjxpack"
ssh -p 6666 -i ~/.ssh/id_rsa jxzhu@172.27.127.191 "rm -r /data/home/jxzhu/.conda/envs/dpgen/lib/python3.9/site-packages/zjxpack"
scp -r -P 6666 -i ~/.ssh/id_rsa ./zjxpack jxzhu@172.27.127.191:/data/home/jxzhu/.conda/envs/work_env/lib/python3.9/site-packages
scp -r -P 6666 -i ~/.ssh/id_rsa ./zjxpack jxzhu@172.27.127.191:/data/home/jxzhu/.conda/envs/dpgen/lib/python3.9/site-packages