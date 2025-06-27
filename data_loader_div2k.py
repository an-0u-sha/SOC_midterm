import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# HR images
!wget -q https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
!unzip -q DIV2K_train_HR.zip -d .

# LR images (bicubic downsampled x4)
!wget -q https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
!unzip -q DIV2K_train_LR_bicubic_X4.zip -d .

class DIV2KDataset(Dataset):
    def __init__(self, root_lr, root_hr, patch_size=128):
        self.lr_files = sorted(os.listdir(root_lr))[:10]
        self.hr_files = sorted(os.listdir(root_hr))[:10]
        self.lr_root = root_lr
        self.hr_root = root_hr
        self.transform = transforms.ToTensor()
        self.patch = patch_size

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr = Image.open(os.path.join(self.lr_root, self.lr_files[idx])).convert('RGB')
        hr = Image.open(os.path.join(self.hr_root, self.hr_files[idx])).convert('RGB')

        # Random crop pair
        w, h = lr.size
        x = torch.randint(0, w - self.patch + 1, size=(1,)).item()
        y = torch.randint(0, h - self.patch + 1, size=(1,)).item()
        lr_c = lr.crop((x, y, x + self.patch, y + self.patch))
        hr_c = hr.crop((x*4, y*4, (x+self.patch)*4, (y+self.patch)*4))

        return self.transform(lr_c), self.transform(hr_c)

if __name__ == '__main__':
    dataset = DIV2KDataset('DIV2K_train_LR_bicubic/X4', 'DIV2K_train_HR')
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for lr_batch, hr_batch in loader:
        print("LR batch shape:", lr_batch.shape, "HR batch shape:", hr_batch.shape)
        break
