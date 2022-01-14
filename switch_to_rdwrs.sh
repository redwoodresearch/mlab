echo $1
ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh $1 "cd ~/mlab; sudo pip install -r requirements.txt"

