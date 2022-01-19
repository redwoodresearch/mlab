#!/bin/bash
echo $1
ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh $1 "pip install web-pdb"
