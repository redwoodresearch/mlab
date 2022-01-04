echo $1
ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh $1 "cd ~/mlab && git remote set-url origin https://github.com/redwoodresearch/mlab"
