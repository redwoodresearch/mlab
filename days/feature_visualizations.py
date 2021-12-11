import argparse
import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import models
from torchvision import transforms


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


standard_transforms = [
    transforms.Pad(12),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(1.2, 1.2), shear=10),
]


color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
color_correlation_normalized = torch.tensor(color_correlation_normalized.T).to(DEVICE)


def linear_decorrelate_color(x):
    return torch.einsum('bcij,cd->bdij', x, color_correlation_normalized)

def to_valid_rgb(image_fn, decorrelate=False):
    def new_image_fn():
        image = image_fn()
        if decorrelate:
            image = linear_decorrelate_color(image)
        return torch.sigmoid(image)
    return new_image_fn


def pixel_params_image(shape, sd=0.01):
    sd = sd or 0.01
    tensor = (torch.randn(*shape) * sd).to(DEVICE).requires_grad_(True)
    return [tensor], lambda: tensor


def freq_params_image(shape, init_std=0.01, decay_power=1):
    batch, channels, height, width = shape

    y_freqs = np.fft.fftfreq(height)[:, None]
    x_freqs = np.fft.fftfreq(width)[:((width+1) // 2) + 1]
    freqs = np.sqrt(x_freqs**2 + y_freqs**2)
    freqs[0, 0] = 1.0 / max(width, height)

    params_shape = (batch, channels, *freqs.shape, 2)
    spectral_params = (init_std * torch.randn(*params_shape)).to(DEVICE).requires_grad_(True)
    unscaled_spectra = torch.view_as_complex(spectral_params)

    scale = 1.0 / freqs ** decay_power
    scale = scale.reshape(1, 1, *scale.shape)
    scale = torch.tensor(scale).float().to(DEVICE)

    def image_fn():
        import torch.fft
        spectra = scale * unscaled_spectra
        return torch.fft.irfftn(spectra, s=(height, width), norm='ortho')

    return [spectral_params], image_fn

def get_params_and_image(
    width,
    height=None,
    init_std=0.01,
    n_batches=1,
    n_channels=3,
    decorrelate=True,
    fft=True,
):
    if height is None:
        height = width

    shape = [n_batches, n_channels, height, width]
    param_f = freq_params_image if fft else pixel_params_image
    params, image_f = param_f(shape, init_std=init_std)
    output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output


def get_layer_channel_loss_fn(layer, channel):
    loss = None

    def layer_hook(model, inputs, outputs):
        nonlocal loss
        loss = (-1) * outputs[:, channel].mean()

    def loss_fn():
        return loss

    layer.register_forward_hook(layer_hook)
    return loss_fn


def tensor_to_img_array(tensor):
    return tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)

def view(tensor):
    image = tensor_to_img_array(tensor)
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def layer_channel_feature_visualization(
    model,
    layer,
    channel,
    n_iterations=20,
    learning_rate=5e-2,
    input_size=(224,224),
):
    model.to(DEVICE).eval()

    params, image_fn = get_params_and_image(*input_size, fft=True, decorrelate=True)
    transforms_fn = transforms.Compose(standard_transforms)
    loss_fn = get_layer_channel_loss_fn(layer=layer, channel=channel)

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    for i in tqdm(range(n_iterations)):
        model(transforms_fn(image_fn()))
        loss = loss_fn()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return image_fn()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-it', '--iterations', type=int, default=20)
    args = parser.parse_args()

    inception3 = models.inception_v3(pretrained=True)
    resnet34 = models.resnet34(pretrained=True)

    layer_channel_entries = [
        (inception3, inception3.Mixed_6b, 476),
        (resnet34, resnet34, 207),
        (resnet34, resnet34.layer2, 0),
    ]

    for model, layer, channel in layer_channel_entries:
        feat_vis = layer_channel_feature_visualization(model, layer, channel, n_iterations=args.iterations)
        view(feat_vis)


if __name__ == "__main__":
    main()
