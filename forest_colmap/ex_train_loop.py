from torch.utils.data import DataLoader
from torchvision import transforms

from colmap_pair_dataset import ColmapPairDataset

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = ColmapPairDataset(
    dataset_dir="dataset",
    transform=transform,
    return_names=True,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

for batch in loader:
    image_a = batch["image_a"]   # [B,3,H,W]
    image_b = batch["image_b"]   # [B,3,H,W]
    T_ab = batch["T_ab"]         # [B,4,4]

    # model input example
    # pred = model(image_a, image_b)
    # loss = loss_fn(pred, T_ab)

    print(image_a.shape, image_b.shape, T_ab.shape)
    break
