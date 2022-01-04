conda deactivate
cd ~;
git clone https://github.com/redwoodresearch/mlab
cd mlab
git reset --hard HEAD
git pull
pip install -r requirements.txt
pip install -e .
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
mkdir ~/working
sudo apt-get install -y ffmpeg #snapd
# sudo snap install code --classic
# code --install-extension ms-python.python
# code --install-extension ms-toolsai.jupyter
mkdir -f ~/mlab_trial

# http://{IP}:8890/lab/tree/mlab_trial?token=eacfafffeb
# http://104.171.200.{EXT}:8890/lab/tree/mlab_trial?token=eacfafffeb
# if something goes wrong with process, use
# ps aux | grep eacfafffeb
# to kill existing process
cd ~;
pkill eacfafffeb;
jupyter nbextension enable --py widgetsnbextension
jupyter lab --collaborative --NotebookApp.token=eacfafffeb --port 8890 --ip 0.0.0.0 &

python -c "import torch; import torchvision.models;import torchtext; import transformers; transformers.AutoModelForCausalLM.from_pretrained('gpt2');transformers.AutoModel.from_pretrained('bert-base-cased');torchvision.models.resnet34(pretrained=True);torchvision.models.resnet50(pretrained=True);torchtext.datasets.WikiText2(split='train');torchtext.datasets.WikiText103(split='train');torchtext.datasets.IMDB(split='train');" &

python -c "import transformers; transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B'); transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')" &

curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
