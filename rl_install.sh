sudo apt-get install -y python-opengl xvfb
mkdir  ~/mlab/days/w3d3
cd ~/mlab/days/w3d3
python3 -m venv rl_env
source rl_env/bin/activate
pip install gym[atari]==0.19.0 pyglet matplotlib
deactivate
