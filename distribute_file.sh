#!/bin/bash
echo $1
scp -i ~/mlab_ssh /mnt/c/Users/taoro/Downloads/model.pt $1:/home/ubuntu/interp_2l_attention_only.pt
