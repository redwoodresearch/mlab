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
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
mkdir ~/working
sudo snap install code
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
python -c "import torch; import torchvision.models;import torchtext; import transformers; transformers.AutoModelForCausalLM.from_pretrained('gpt2');transformers.AutoModelForCausalLM.from_pretrained('bert-base-cased');torchvision.models.resnet34(pretrained=True);torchvision.models.resnet50(pretrained=True);torchtext.datasets.WikiText2(split='train');torchtext.datasets.WikiText103(split='train');torchtext.datasets.IMDB(split='train');"