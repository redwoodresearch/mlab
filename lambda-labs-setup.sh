# TODO: write this
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
conda deactivate
cd ~;
git clone https://github.com/taoroalin/mlab
cd mlab
pip install -r requirements.txt
pip install -e .
mkdir ~/working
sudo snap install code --classic
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter