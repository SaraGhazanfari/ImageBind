import sys
import torch
from torch import nn
from tqdm import tqdm
from data.night_dataset import NightDataset
from imagebind import ModalityType


@torch.no_grad()
def dreamsim_eval(model, root_dir, batch_size, device):
    data_loader, dataset_size = NightDataset(root_dir=root_dir, batch_size=batch_size,
                                             split='test_imagenet').get_dataloader()
    no_imagenet_data_loader, no_imagenet_dataset_size = NightDataset(root_dir=root_dir,
                                                                     batch_size=batch_size,
                                                                     split='test_no_imagenet').get_dataloader()
    print(len(data_loader), len(no_imagenet_data_loader))
    imagenet_score = get_2afc_score_eval(model, data_loader, device)
    print(f"ImageNet 2AFC score: {str(imagenet_score)}")
    torch.cuda.empty_cache()
    no_imagenet_score = get_2afc_score_eval(model, no_imagenet_data_loader, device)
    print(f"No ImageNet 2AFC score: {str(no_imagenet_score)}")
    overall_score = (imagenet_score * dataset_size + no_imagenet_score * no_imagenet_dataset_size) / (
            dataset_size + no_imagenet_dataset_size)
    print(f"Overall 2AFC score: {str(overall_score)}")


def one_step_2afc_score_eval(model, img_ref, img_left, img_right, target):
    dist_0, dist_1, _ = get_cosine_score_between_images(model, img_ref, img_left, img_right)
    if len(dist_0.shape) < 1:
        dist_0 = dist_0.unsqueeze(0)
        dist_1 = dist_1.unsqueeze(0)
    dist_0 = dist_0.unsqueeze(1)
    dist_1 = dist_1.unsqueeze(1)
    target = target.unsqueeze(1)
    return dist_0, dist_1, target


def get_2afc_score(d0s, d1s, targets):
    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores)
    return twoafc_score


def get_2afc_score_eval(model, test_loader, device):
    print("Evaluating NIGHTS dataset.")
    d0s = []
    d1s = []
    targets = []
    # with torch.no_grad()
    for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img_ref, img_left, img_right, target = img_ref.to(device), img_left.to(device), \
            img_right.to(device), target.to(device)
        dist_0, dist_1, target = one_step_2afc_score_eval(model, img_ref, img_left, img_right, target)
        d0s.append(dist_0)
        d1s.append(dist_1)
        targets.append(target)

    twoafc_score = get_2afc_score(d0s, d1s, targets)
    return twoafc_score


def get_cosine_score_between_images(model, img_ref, img_left, img_right):
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    embed_ref = model({ModalityType.VISION: img_ref})[ModalityType.VISION]
    embed_x0 = model({ModalityType.VISION: img_left})[ModalityType.VISION]
    embed_x1 = model({ModalityType.VISION: img_right})[ModalityType.VISION]
    bound = torch.norm(embed_x0 - embed_x1, p=2, dim=(1)).unsqueeze(1)
    dist_0 = 1 - cos_sim(embed_ref, embed_x0)
    dist_1 = 1 - cos_sim(embed_ref, embed_x1)
    return dist_0, dist_1, bound
