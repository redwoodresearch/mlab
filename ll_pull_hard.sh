#!/usr/bin/env bash

echo $1
ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh $1 "cd ~/mlab && git stash && git remote set-url origin https://github.com/redwoodresearch/mlab && git checkout -f origin/main && wall 'UPDATED GIT REPO'"
