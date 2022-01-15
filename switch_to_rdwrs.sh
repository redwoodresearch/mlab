echo $1
ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh $1 "pip install traitlets==5.1.1; cd ~;jupyter lab --collaborative --NotebookApp.token=eacfafffeb --port 8890 --ip 0.0.0.0 &"
# ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh $1 "cd ~/mlab; sudo apt-get install -y python3-venv; python -m venv mlab_env; pip install -r requirements.txt; pip install traitlets==5.1.1; cd ~;jupyter lab --collaborative --NotebookApp.token=eacfafffeb --port 8890 --ip 0.0.0.0 &"


