import torch.nn as nn
import torch


class FCDiscriminator(nn.Module):
	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()
		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)

		return x

	
class DHA(nn.Module):
	def __init__(self, discriminator):
		super(DHA, self).__init__()
		self.discriminator = discriminator

	def forward(self, target, source, loss_type):
		target_feat = self.discriminator(target)
		source_feat = self.discriminator(source)

		if loss_type == 'adversarial':
			loss = torch.nn.ReLU()(torch.mean(source_feat) - torch.mean(target_feat))
		elif loss_type == 'discriminator':
			loss = (torch.mean(torch.nn.ReLU()(1 - source_feat)) +
					torch.mean(torch.nn.ReLU()(1 + target_feat)))
		else:
			NotImplementedError("Unavailable loss type!")
		return loss
