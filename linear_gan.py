from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import os

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(504, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, time_series):
        return self.discriminator(time_series)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 504)
        )

    def forward(self, noise):
        return self.generator(noise)


class LinearGAN():

    def __init__(self, input_data, epochs=5000, lambda_gp=5, generator_path='generator.pth', discriminator_path='discriminator.pth'):

        self.cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.generator_path = generator_path
        self.discriminator_path = discriminator_path

        self.generator, self.optimizer_generator = self._init_generator_(
            generator_path)
        self.discriminator, self.optimizer_discriminator = self._init_discriminator_(
            discriminator_path)

        self.last_epoch_saved = self._get_last_epoch_(generator_path)

        self.lambda_gp = lambda_gp
        self.epochs = epochs
        self.noise_dim = 32
        self.batch_size = 256

        self.dataloader = self._get_tesor_(input_data)

    def _get_tesor_(self, input_data):
        input_tensor = torch.tensor(input_data.T.values).to(self.cuda)

        means = input_tensor.mean(0, keepdim=True)
        deviations = input_tensor.std(0, keepdim=True)

        input_tensor_scaled = (input_tensor - means) / (deviations + 0.000001)

        dataloader = torch.utils.data.DataLoader(
            input_tensor_scaled, batch_size=self.batch_size)

        assert input_tensor_scaled.shape[1] == 504

        return dataloader

    def _init_generator_(self, model_path):    

        generator = Generator().to(self.cuda)
        optimizer_generator = torch.optim.Adam(generator.parameters())

        if os.path.exists(model_path):

            print('initializing generator')

            checkpoint_generator = torch.load(model_path)
            generator.load_state_dict(checkpoint_generator['model_state_dict'])
            optimizer_generator.load_state_dict(
                checkpoint_generator['optimizer_state_dict'])

        return (generator, optimizer_generator)

    def _init_discriminator_(self, model_path):       

        discriminator = Discriminator().to(self.cuda)
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters())

        if os.path.exists(model_path):

            print('initializing discriminator')

            checkpoint_discriminator = torch.load(model_path)
            discriminator.load_state_dict(
                checkpoint_discriminator['model_state_dict'])
            optimizer_discriminator.load_state_dict(
                checkpoint_discriminator['optimizer_state_dict'])

        return (discriminator, optimizer_discriminator)

    def _get_last_epoch_(self, model_path='generator.pth'):

        last_epoch_saved = 0

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            last_epoch_saved = checkpoint['epoch']

        return last_epoch_saved

    def _compute_gradient_penalty_(self, discriminator, real_samples, fake_samples, batch_size):

        alpha = self.Tensor(np.random.normal(
            0, 1, (batch_size, 504))).unsqueeze(0)

        interpolates = (alpha * real_samples + ((1 - alpha)
                                                * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        fake = Variable(self.Tensor(1, batch_size, 1).fill_(
            1.0), requires_grad=False)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def _train_report_(self, epoch, batch, discriminator_loss, generator_loss):

        show_train_step = epoch % 50 == 0 and batch == 0
        if show_train_step:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, self.epochs, batch, len(self.dataloader), discriminator_loss.item(), generator_loss.item())
            )

        show_generated_time_serie = epoch % 100 == 0 and batch == 0
        if show_generated_time_serie:
            noise = Variable(self.Tensor(np.random.normal(
                0, 1, (self.batch_size, self.noise_dim))))
            fake_ts = self.generator.forward(noise.unsqueeze(0))
            plt.plot(fake_ts.cpu().detach().numpy().squeeze()[0])
            plt.show()

    def _save_model_(self, epoch, batch, generator_loss, discriminator_loss):
        will_save_model = epoch % 500 == 0 and epoch != 0 and batch == 0
        if will_save_model:
            print('Saving model')

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.generator.state_dict(),
                'optimizer_state_dict': self.optimizer_generator.state_dict(),
                'loss': generator_loss,
            }, self.generator_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer_discriminator.state_dict(),
                'loss': discriminator_loss,
            }, self.discriminator_path)

    def train(self):
        for epoch in tqdm(range(self.last_epoch_saved, self.epochs)):
            for batch, time_serie in enumerate(self.dataloader):

                batch_size_epoch = time_serie.shape[0]
                real_time_serie = time_serie

                self.optimizer_discriminator.zero_grad()

                noise = Variable(self.Tensor(np.random.normal(
                    0, 1, (batch_size_epoch, self.noise_dim)))).to(self.cuda)

                fake_time_serie = self.generator(noise.unsqueeze(0))

                fake_validity = self.discriminator(fake_time_serie.float())
                real_validity = self.discriminator(
                    real_time_serie.unsqueeze(0).float())

                gradient_penalty = self._compute_gradient_penalty_(
                    self.discriminator, real_validity, fake_validity, batch_size_epoch)

                discriminator_loss = -torch.mean(real_validity) + torch.mean(
                    fake_validity) + self.lambda_gp * gradient_penalty

                discriminator_loss.backward()
                self.optimizer_discriminator.step()

                self.optimizer_generator.zero_grad()

                will_train_generator = batch % 10 == 0
                if will_train_generator:

                    fake_time_serie = self.generator(noise.unsqueeze(0))

                    fake_validity = self.discriminator(fake_time_serie.float())

                    generator_loss = -torch.mean(fake_validity)

                    generator_loss.backward()
                    self.optimizer_generator.step()

                    self._train_report_(
                        epoch, batch, discriminator_loss, generator_loss)
                    self._save_model_(
                        epoch, batch, generator_loss, discriminator_loss)
