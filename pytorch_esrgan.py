import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# RRDB block (simplified)
class RRDB(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out = self.conv3(out)
        return x + out

class Generator(nn.Module):
    def __init__(self, num_rrdb=5):
        super().__init__()
        self.initial = nn.Conv2d(3, 64, 3, padding=1)
        self.rrdbs = nn.Sequential(*[RRDB(64) for _ in range(num_rrdb)])
        self.final = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        feat = self.initial(x)
        out = self.rrdbs(feat)
        return self.final(out)

def load_image(path):
    img = Image.open(path).convert('RGB')
    preprocess = transforms.ToTensor()
    return preprocess(img).unsqueeze(0)

def save_image(tensor, path):
    img = tensor.squeeze().clamp(0, 1)
    save = transforms.ToPILImage()(img)
    save.save(path)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator().to(device)
    # Assume model weights loaded 
    # model.load_state_dict(torch.load('generator.pth'))

    lr = load_image('/content/1b3b0a5e1ab348ccae48a148b7edb167.png').to(device)
    with torch.no_grad():
        sr = model(lr)
import os
os.makedirs("outputs", exist_ok=True)  

save_image(sr.cpu(), 'outputs/sr.png')
print("Super-resolved image saved to outputs/sr.png")
