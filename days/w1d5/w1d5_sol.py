import argparse
import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import models
from torchvision import transforms
from days.utils import tpeek
import time
import random

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

standard_transforms = [
    transforms.Pad(12),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.RandomAffine(
        degrees=10, translate=(0.05, 0.05), scale=(1.2, 1.2), shear=10
    ),
]

color_correlation_svd_sqrt = np.asarray(
    [
        [0.26, 0.09, 0.02],
        [0.27, 0.00, -0.05],
        [0.27, -0.09, 0.03],
    ]
).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
color_correlation_normalized = torch.tensor(color_correlation_normalized.T).to(DEVICE)


def linear_decorrelate_color(x):
    return torch.einsum("bcij,cd->bdij", x, color_correlation_normalized)


def to_valid_rgb(image_fn, decorrelate=False):
    def new_image_fn():
        image = image_fn()
        if decorrelate:
            image = linear_decorrelate_color(image)
        result = torch.sigmoid(image)
        return result

    return new_image_fn


def pixel_params_image(shape, sd=0.01, **kwargs):
    sd = sd or 0.01
    tensor = (torch.randn(*shape) * sd).to(DEVICE).requires_grad_(True)
    return [tensor], lambda: tensor


def freq_params_image(shape, init_std=0.01, decay_power=1):
    batch, channels, height, width = shape

    y_freqs = np.fft.fftfreq(height)[:, None]  # (224,) (224,1)
    x_freqs = np.fft.fftfreq(width)[: ((width + 1) // 2) + 1]  # (113,)
    freqs = np.sqrt(x_freqs ** 2 + y_freqs ** 2)
    freqs[0, 0] = 1.0 / max(width, height)
    params_shape = (batch, channels, *freqs.shape, 2)
    spectral_params = (
        (init_std * torch.randn(*params_shape)).to(DEVICE).requires_grad_(True)
    )
    unscaled_spectra = torch.view_as_complex(spectral_params)

    scale = 1.0 / freqs ** decay_power
    scale = scale.reshape(1, 1, *scale.shape)
    scale = torch.tensor(scale).float().to(DEVICE)

    def image_fn():
        import torch.fft

        spectra = scale * unscaled_spectra
        return torch.fft.irfftn(spectra, s=(height, width), norm="ortho")

    return [spectral_params], image_fn


def get_params_and_image(
    height,
    width=None,
    init_std=0.01,
    n_batches=1,
    n_channels=3,
    decorrelate=True,
    fft=True,
):
    if width is None:
        width = height

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


def view(tensor, name="the_image"):
    image = tensor_to_img_array(tensor)
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    # Image.fromarray(image).show(name)
    Image.fromarray(image).save(name)


def layer_channel_feature_visualization(
    model,
    layer,
    channel,
    n_iterations=20,
    learning_rate=5e-2,
    input_size=(224, 224),
    fft=True,
    decorrelate=True,
    use_transforms=True,
):
    model.to(DEVICE).eval()

    params, image_fn = get_params_and_image(
        *input_size, fft=fft, decorrelate=decorrelate
    )
    transforms_fn = transforms.Compose(standard_transforms if use_transforms else [])
    loss_fn = get_layer_channel_loss_fn(layer=layer, channel=channel)

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    for i in tqdm(range(n_iterations)):
        model(transforms_fn(image_fn()))
        loss = loss_fn()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time.sleep(0.03)
    return image_fn()


def get_layer_activations_fn(model, layer):
    activations = None

    def hook_fn(model, input, output):
        nonlocal activations
        activations = output

    layer.register_forward_hook(hook_fn)

    def activations_fn(input):
        model(input)
        return activations

    return activations_fn


def activation_match(model, layer, target, n_iterations=50, learning_rate=5e-2):
    model.to(DEVICE).eval()

    # get target activations
    activations_fn = get_layer_activations_fn(model, layer)

    target_activations = activations_fn(target).detach()
    input_size = target.size()[-2:]
    params, image_fn = get_params_and_image(*input_size, fft=True, decorrelate=True)
    transforms_fn = transforms.Compose(
        [*standard_transforms, transforms.Resize(target.shape[-2:])]
    )

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    for i in tqdm(range(n_iterations)):
        out_activations = activations_fn(transforms_fn(image_fn()))
        loss = torch.mean((target_activations - out_activations) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return image_fn()


def feature_visualization_example1(args):
    # inception3 = models.inception_v3(pretrained=True)
    # inception1 = models.googlenet(pretrained=True)
    # resnet34 = models.resnet34(pretrained=True)
    # from robustness.model_utils import make_and_restore_model
    # import torch as ch
    # from robustness.datasets import ImageNet
    # ds = ImageNet("/")
    # resnet50, _ = make_and_restore_model(
    #     arch="resnet50", dataset=ds, state_dict_path="resnet50_madry.pt"
    # )

    resnet50 = torch.load("resnet50_madry")
    permuted_channels = list(range(0, 1000))
    random.shuffle(permuted_channels)
    layer_channel_entries = [(resnet50, resnet50, x) for x in permuted_channels[:100]]
    # layer_channel_entries = [(resnet34, resnet34, 50)]
    # layer_channel_entries = [(resnet50, resnet50, x) for x in range(20)]
    for model, layer, channel in layer_channel_entries:
        print("CHANNEL", channel)
        feat_vis = layer_channel_feature_visualization(
            model,
            layer,
            channel,
            n_iterations=args.iterations,
            decorrelate=True,
            fft=True,
            use_transforms=True,
        )
        view(feat_vis, "image_madry" + str(channel) + ".png")
        time.sleep(2)


def feature_visualization_layers_grid(args):
    import matplotlib.pyplot as plt
    from torchvision.models.resnet import BasicBlock, Bottleneck

    resnet = models.resnet34(pretrained=True)
    is_residual_block = lambda m: isinstance(m, BasicBlock) or isinstance(m, Bottleneck)
    residual_blocks = [m for m in resnet.modules() if is_residual_block(m)]
    layers = residual_blocks + [resnet]
    n_channels = 10

    fig_width_per_col = 3
    fig_height_per_row = 3
    n_rows = len(layers)
    n_cols = n_channels
    figsize = (fig_width_per_col * n_cols, fig_height_per_row * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i, layer in enumerate(layers):
        for channel in range(n_channels):
            feat_vis = layer_channel_feature_visualization(
                resnet, layer, channel, n_iterations=args.iterations
            )
            feat_vis_img = tensor_to_img_array(feat_vis)[0]
            axes[i, channel].imshow(feat_vis_img)
            axes[i, channel].set_axis_off()

    plt.savefig("out.png")
    plt.show()


def match_activations_example(args):
    import matplotlib.pyplot as plt
    from PIL import Image
    import requests
    from io import BytesIO

    def load_image(url):
        response = requests.get(url)
        return Image.open(BytesIO(response.content))

    url = "https://i.redd.it/1qyooc1sekb61.jpg"
    img = load_image(url)

    input = transforms.Resize((224, 224))(
        transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    )
    print(input.shape)
    resnet = models.resnet34(pretrained=True)
    # resnet = torch.load("resnet50_madry")

    # print(resnet)
    arg_layers = [
        resnet.bn1,
        # resnet.conv2,
        resnet.layer1[0],
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        resnet,
    ]
    fig, axes = plt.subplots(1, len(arg_layers), figsize=(30, 6))
    axes = axes.flatten()
    axes[0].imshow(input.cpu()[0].permute(1, 2, 0))
    axes[0].set_axis_off()
    for i, layer in enumerate(arg_layers, 1):
        vis = activation_match(
            model=resnet, layer=layer, target=input, n_iterations=args.iterations
        )
        view(vis, "image_match" + str(i) + ".png")
        feat_vis_img = tensor_to_img_array(vis)[0]
        axes[i].imshow(feat_vis_img)
        axes[i].set_axis_off()
    plt.show()
    plt.savefig("match_sbs.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--iterations", type=int, default=70)
    parser.add_argument("-e", "--example", type=int, default=0)
    args = parser.parse_args()

    example_fns = [
        feature_visualization_example1,
        feature_visualization_layers_grid,
        match_activations_example,
    ]

    example_fns[args.example](args)
