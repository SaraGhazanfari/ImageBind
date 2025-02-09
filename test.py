import argparse
import os

from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from eval_2afc import dreamsim_eval
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from advertorch.attacks import L2PGDAttack, LinfPGDAttack
import torch.nn as nn
from torchvision import transforms as pth_transforms


def generate_attack_to_imagebind(model, device, cos_sim):
    image_paths = [".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]

    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }
    # adversary = L2PGDAttack(model, eps=1.0, loss_fn=nn.MSELoss(), nb_iter=100, eps_iter=0.01,
    #                         rand_init=True,
    #                         clip_min=0., clip_max=1., targeted=False)
    adversary = LinfPGDAttack(model, loss_fn=nn.MSELoss(), eps=0.05, nb_iter=50,
                              eps_iter=0.03, rand_init=True, clip_min=0., clip_max=1.,
                              targeted=False)
    adv_image = adversary(inputs, model(inputs))
    with torch.no_grad():
        embeddings = model(inputs)
        adv_embeddings = model({ModalityType.VISION: adv_image})
        print(cos_sim(embeddings[ModalityType.VISION][0:], adv_embeddings[ModalityType.VISION][0:]))


def load_data(data_path, batch_size):
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
    ])
    dataset_val = ImageFolder(os.path.join(data_path, "val"), transform=transform)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )
    img_ref = transform(Image.open('ref.jpg', mode='r'))
    return data_loader_val, img_ref


def get_distance_from_texts(model, cos_sim):
    text_list = ["An apple in a garden filled with flowers.",
                 "A small class in the girl's high school that has an apple in desk"]

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    }
    with torch.no_grad():
        embeddings = model(inputs)[ModalityType.TEXT]
        print(cos_sim(embeddings[0:-1], embeddings[-1:]))


@torch.no_grad()
def get_distance_within_images(img_embed, val_loader, cos_sim, model, device):
    distance_list = list()
    for images, _ in tqdm(val_loader):
        embed = model({ModalityType.VISION: images.to(device)})[ModalityType.VISION]
        distance_list.extend(cos_sim(img_embed, embed[0:]))
    torch.save(distance_list, 'inter_class_distance_list.pt')


@torch.no_grad()
def get_the_2afc_score(model, data_dir, batch_size, device):
    dreamsim_eval(model, data_dir, batch_size, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--data_path', default='/imagenet', type=str)

    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    # generate_attack_to_imagebind(model, device, cos_sim)
    # val_loader, img_ref = load_data(data_path=args.data_path, batch_size=args.batch_size)
    # get_distance_within_images(img_embed, val_loader, cos_sim, model, device)
    get_the_2afc_score(model=model, data_dir=args.data_path, batch_size=args.batch_size, device=device)
