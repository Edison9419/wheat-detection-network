# coding=utf-8
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
num_epoch = 25
z_dimension = 100

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.dis(x)
        return x

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dimension+10, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.gen(x)
        return x

D = discriminator()
G = generator()
D = D.to(device)
G = G.to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(num_epoch):
    for i, (img, label) in enumerate(dataloader):
        num_img = img.size(0)
        label_onehot = torch.zeros((num_img,10)).to(device)
        label_onehot[torch.arange(num_img),label]=1

        img = img.view(num_img,  -1)
        real_img = img.to(device)
        real_label = label_onehot
        fake_label = torch.zeros((num_img,10)).to(device)

        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)

        z = torch.randn(num_img, z_dimension+10).to(device)
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        z = torch.randn(num_img, z_dimension).to(device)
        z = torch.cat([z, real_label],1)
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '.format(
                epoch, num_epoch, d_loss.item(), g_loss.item(),
            ))

        if epoch == 0:
            real_images = real_img.cpu().clamp(0,1).view(-1,1,28,28).data
            save_image(real_images, './img_CGAN/real_images.png')
        if i == len(dataloader)-1:
            fake_images = fake_img.cpu().clamp(0,1).view(-1,1,28,28).data
            save_image(fake_images, './img_CGAN/fake_images-{}.png'.format(epoch + 1))
