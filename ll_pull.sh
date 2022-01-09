#!/usr/bin/env bash

echo $1
ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh $1 "cd ~/mlab && git pull && pip install -e ."
