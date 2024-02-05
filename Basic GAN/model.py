import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # To make all value between -1 and 1
        )

    def forward(self, x):
        return self.gen(x)

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 3e-4
Z_DIM = 64
IMAGE_DIM = 28 * 28
BATCH_SIZE = 32
EPOCHS = 50

disc = Discriminator(img_dim=IMAGE_DIM).to(device=DEVICE)
gen = Generator(Z_DIM, IMAGE_DIM).to(DEVICE)
fixed_noise = torch.rand((BATCH_SIZE, Z_DIM)).to(DEVICE)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
datasets = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=LR)
opt_gen = optim.Adam(gen.parameters(), lr=LR)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(DEVICE)
        batch_size = real.shape[0]

        # Train Discriminator: max log(D(real)) + log(1 - D(G(z))
        # This is to obtain max log(D(real)), actually -min log(D(real))
        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        # Now log(1 - D(G(z))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        # Now put together
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator min log(1 - D(G(z))) <--> max log(D(G(z))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
