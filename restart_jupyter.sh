echo $1
ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh $1 "cd ~/mlab; pkill eacfafffeb; jupyter lab --collaborative --NotebookApp.token=eacfafffeb --port 8890 --ip 0.0.0.0 &"
curl "http://$1:8890/lab/workspaces/lab?token=eacfafffeb&reset"
