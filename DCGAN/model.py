import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Discriminator(nn.Module):
    def __init__(self, img_channels, channels_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=channels_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=channels_d, out_channels=channels_d*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_d*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=channels_d*2, out_channels=channels_d*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_d*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=channels_d*4, out_channels=channels_d*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_d*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=channels_d*8, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels, channels_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_channels, out_channels=channels_g*16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(channels_g*16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels_g*16, out_channels=channels_g*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_g*8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels_g*8, out_channels=channels_g*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_g*4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels_g*4, out_channels=channels_g*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels_g*2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels_g*2, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
batch_size = 64
img_size = 64
channel_img = 1
channel_noise = 256
epochs = 10

channels_d = 16
channels_g = 16

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
datasets = datasets.MNIST(root="../dataset/", transform=transform, download=True)
loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

disc = Discriminator(channel_img, channels_d).to(device)
gen = Generator(channel_noise, channel_img, channels_g).to(device)

opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

disc.train()
gen.train()

fixed_noise = torch.randn(64, channel_noise, 1, 1).to(device)
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(epochs):
    loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]")
    for batch_idx, (data, _) in enumerate(loop):
        data = data.to(device)
        batch_size = data.shape[0]

        # Train Discirminator : max log(D(x)) + log(1 - D(G(z)))
        # we use a trick where first we train on all true image and then all fake,
        # seems to improve training
        disc.zero_grad()
        label = (torch.ones(batch_size)*0.9).to(device)  # 0.9 is a trick to improve training
        output = disc(data).reshape(-1)  # here we send the real data
        lossD_real = criterion(output, label)
        #D_x = output.mean().item()

        noise = torch.randn(batch_size, channel_noise, 1, 1).to(device)
        fake = gen(noise)
        label = (torch.ones(batch_size)*0.1).to(device)  # 0.1 same trick to improve training
        # We use detach to don't consider the gradient of the fake
        output = disc(fake.detach()).reshape(-1)  # here we send all fake data
        lossD_fake = criterion(output, label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        opt_disc.step()

        # Train Generator: max log(D(G(z))), easier gradient than the classic formula
        gen.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = disc(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        opt_gen.step()

        if batch_idx % 200 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise)

                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

        # tqdm loop
        loop.set_postfix(lossD=lossD.item(), lossG=lossG.item())
