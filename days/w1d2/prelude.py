# We first load a pretrained ResNet model from the PyTorch vision repository.

from torchvision import models
torchvision_resnet34 = models.resnet34(pretrained=True)
resnet34 = torchvision_resnet34
_ = resnet34.eval()

# We will download images from the internet to feed into the model.

from PIL import Image
import requests
from io import BytesIO

def load_image(url):
  response = requests.get(url)
  return Image.open(BytesIO(response.content))

url = "https://www.oregonzoo.org/sites/default/files/styles/article-full/public/animals/H_chimpanzee%20Jackson.jpg"
img = load_image(url)

# The model expects a batch of images of shape `(num_images, num_channels=3, height, width)`,
# and outputs the classfication logits of shape `(num_images, num_classes=1000)`.

import torch
from torchvision import transforms

inputs = transforms.ToTensor()(img).unsqueeze_(0)
print(inputs.shape)
outputs = resnet34(inputs)
outputs = outputs.reshape(-1)
print(outputs.shape)

# We load the ImageNet class labels.

imagenet_labels_url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
response = requests.get(imagenet_labels_url)
imagenet_labels = eval(response.text)

# We select a few random images from the internet, and check the classification results from the model,
# by looking at the most likely and least likely predicted classes.

urls = [
  "https://www.oregonzoo.org/sites/default/files/styles/article-full/public/animals/H_chimpanzee%20Jackson.jpg",
  "https://anipassion.com/ow_userfiles/plugins/animal/breed_image_56efffab3e169.jpg",
  "https://upload.wikimedia.org/wikipedia/commons/f/f2/Platypus.jpg",
  "https://static5.depositphotos.com/1017950/406/i/600/depositphotos_4061551-stock-photo-hourglass.jpg",
  "https://img.nealis.fr/ptv/img/p/g/1465/1464424.jpg",
  "http://www.tudobembresil.com/wp-content/uploads/2015/11/nouvelancopacabana.jpg",
  "https://ychef.files.bbci.co.uk/976x549/p0639ffn.jpg",
  "https://www.thoughtco.com/thmb/Dk3bE4x1qKqrF6LBf2qzZM__LXE=/1333x1000/smart/filters:no_upscale()/iguana2-b554e81fc1834989a715b69d1eb18695.jpg",
  "https://i.redd.it/mbc00vg3kdr61.jpg",
  "https://static.wikia.nocookie.net/disneyfanon/images/a/af/Goofy_pulling_his_ears.jpg",
]

def show_classes_probabilities(k=3):
  for url in urls:
    img = load_image(url)

    inputs = transforms.ToTensor()(img).unsqueeze_(0)
    outputs = resnet34(inputs)
    probs = torch.softmax(outputs, -1).flatten()
    sorted_probs, sorted_idxs = probs.sort(descending=True)
    sorted_probs = [p.item() for p in sorted_probs]
    sorted_classes = [imagenet_labels[idx.item()] for idx in sorted_idxs]

    small_img = img.copy()
    small_img.thumbnail((150, 150))
    display(small_img)
    print(
      *(f'{100*prob:05.2f}% | {label}' for prob, label in zip(sorted_probs[:k], sorted_classes[:k])),
      '...',
      *(f'{100*prob:08.5f}% | {label}' for prob, label in zip(sorted_probs[-k:], sorted_classes[-k:])),
      sep='\n',
      end='\n\n',
    )

show_classes_probabilities()
