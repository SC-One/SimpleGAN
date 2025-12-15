import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import os
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
z_dim = 100
lr = 2e-4
epochs = 50
save_dir = "gan_outputs"
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # scale to [-1, 1]
])
trainset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_dim),
            nn.Tanh()  # because images normalized to [-1, 1]
        )

    def forward(self, z):
        img = self.net(z)
        return img.view(z.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = img.view(img.size(0), -1)
        return self.net(x).view(-1)

G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, z_dim, device=device)

print("Starting training on device:", device)
for epoch in range(1, epochs + 1):
    g_loss_running = 0.0
    d_loss_running = 0.0
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)

        batch_size_curr = imgs.size(0)
        real_labels = torch.ones(batch_size_curr, device=device)
        fake_labels = torch.zeros(batch_size_curr, device=device)

        # Real images
        D.zero_grad()
        out_real = D(imgs)
        loss_real = criterion(out_real, real_labels)

        # Fake images
        noise = torch.randn(batch_size_curr, z_dim, device=device)
        fake_imgs = G(noise).detach()  # detach so G not updated on D step
        out_fake = D(fake_imgs)
        loss_fake = criterion(out_fake, fake_labels)

        d_loss = loss_real + loss_fake
        d_loss.backward()
        opt_D.step()

        G.zero_grad()
        noise = torch.randn(batch_size_curr, z_dim, device=device)
        gen_imgs = G(noise)
        preds = D(gen_imgs)
        g_loss = criterion(preds, real_labels)
        g_loss.backward()
        opt_G.step()

        g_loss_running += g_loss.item()
        d_loss_running += d_loss.item()

    avg_g = g_loss_running / len(loader)
    avg_d = d_loss_running / len(loader)
    print(f"Epoch [{epoch}/{epochs}]  D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}")

    with torch.no_grad():
        samples = G(fixed_noise).cpu()
        samples = (samples + 1.0) / 2.0  # rescale to [0,1] for saving
        utils.save_image(samples, os.path.join(save_dir, f"epoch_{epoch:03d}.png"), nrow=8)

print("Training finished. Generated images saved in:", save_dir)
