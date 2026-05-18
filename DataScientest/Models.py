"""Model zoo for P3SL split-learning experiments.

Provides VGG-family and ResNet-family architectures wrapped so that each
model can be sliced at an arbitrary layer into a client-side sub-model and a
server-side sub-model. The public factory is :func:`getModels`, which
returns either a single full model or the (client, server) pair given a
split-layer index ``sl``.
"""

import collections, torch, torchvision, os
import torch.nn as nn

class MnistNet(nn.Module):
	def __init__(self, n_channels=1):
		super(MnistNet, self).__init__()
		self.features = []
		self.initial = None
		self.classifier = []
		self.layers = collections.OrderedDict()

		self.conv1 = nn.Conv2d(
			in_channels=n_channels,
			out_channels=8,
			kernel_size=5
		)
		self.features.append(self.conv1)
		self.layers['conv1'] = self.conv1

		self.ReLU1 = nn.ReLU(False)
		self.features.append(self.ReLU1)
		self.layers['ReLU1'] = self.ReLU1

		self.pool1 = nn.MaxPool2d(2, 2)
		self.features.append(self.pool1)
		self.layers['pool1'] = self.pool1

		self.conv2 = nn.Conv2d(
			in_channels=8,
			out_channels=16,
			kernel_size=5
		)
		self.features.append(self.conv2)
		self.layers['conv2'] = self.conv2

		self.ReLU2 = nn.ReLU(False)
		self.features.append(self.ReLU2)
		self.layers['ReLU2'] = self.ReLU2

		self.pool2 = nn.MaxPool2d(2, 2)
		self.features.append(self.pool2)
		self.layers['pool2'] = self.pool2

		self.feature_dims = 16 * 4 * 4
		self.fc1 = nn.Linear(self.feature_dims, 120)
		self.classifier.append(self.fc1)
		self.layers['fc1'] = self.fc1

		self.fc1act = nn.ReLU(False)
		self.classifier.append(self.fc1act)
		self.layers['fc1act'] = self.fc1act

		self.fc2 = nn.Linear(120, 84)
		self.classifier.append(self.fc2)
		self.layers['fc2'] = self.fc2

		self.fc2act = nn.ReLU(False)
		self.classifier.append(self.fc2act)
		self.layers['fc2act'] = self.fc2act

		self.fc3 = nn.Linear(84, 10)
		self.classifier.append(self.fc3)
		self.layers['fc3'] = self.fc3

		self.initial_params = [param.clone().detach().data for param in self.parameters()]

	def getName(self):
		return "MnistNet"

	def forward(self, x, start=0, end=10):
		if start <= 5:  # start in self.features
			for idx, layer in enumerate(self.features[start:]):
				x = layer(x)
				if idx == end:
					return x
			x = x.view(-1, self.feature_dims)
			for idx, layer in enumerate(self.classifier):
				x = layer(x)
				if idx + 6 == end:
					return x
			return x
		else:
			if start == 6:
				x = x.view(-1, self.feature_dims)
			for idx, layer in enumerate(self.classifier):
				if idx >= start - 6:
					x = layer(x)
				if idx + 6 == end:
					return x

	def get_params(self, end=10):
		params = []
		for layer in list(self.layers.values())[:end + 1]:
			params += list(layer.parameters())
		return params

	def restore_initial_params(self):
		for param, initial in zip(self.parameters(), self.initial_params):
			param.data = initial.requires_grad_(True)


class CifarNet(nn.Module):
	def __init__(self, n_channels=1, n_class=10):
		super(CifarNet, self).__init__()
		self.features = []
		self.initial = None
		self.classifier = []
		self.layers = collections.OrderedDict()

		self.conv11 = nn.Conv2d(
			in_channels=n_channels,
			out_channels=64,
			kernel_size=3,
			padding=1
		)
		self.features.append(self.conv11)
		self.layers['conv11'] = self.conv11

		self.ReLU11 = nn.ReLU(True)
		self.features.append(self.ReLU11)
		self.layers['ReLU11'] = self.ReLU11

		self.conv12 = nn.Conv2d(
			in_channels=64,
			out_channels=64,
			kernel_size=3,
			padding=1
		)
		self.features.append(self.conv12)
		self.layers['conv12'] = self.conv12

		self.ReLU12 = nn.ReLU(True)
		self.features.append(self.ReLU12)
		self.layers['ReLU12'] = self.ReLU12

		self.pool1 = nn.MaxPool2d(2, 2)
		self.features.append(self.pool1)
		self.layers['pool1'] = self.pool1

		self.conv21 = nn.Conv2d(
			in_channels=64,
			out_channels=128,
			kernel_size=3,
			padding=1
		)
		self.features.append(self.conv21)
		self.layers['conv21'] = self.conv21

		self.ReLU21 = nn.ReLU(True)
		self.features.append(self.ReLU21)
		self.layers['ReLU21'] = self.ReLU21

		self.conv22 = nn.Conv2d(
			in_channels=128,
			out_channels=128,
			kernel_size=3,
			padding=1
		)
		self.features.append(self.conv22)
		self.layers['conv22'] = self.conv22

		self.ReLU22 = nn.ReLU(True)
		self.features.append(self.ReLU22)
		self.layers['ReLU22'] = self.ReLU22

		self.pool2 = nn.MaxPool2d(2, 2)
		self.features.append(self.pool2)
		self.layers['pool2'] = self.pool2

		self.conv31 = nn.Conv2d(
			in_channels=128,
			out_channels=128,
			kernel_size=3,
			padding=1
		)
		self.features.append(self.conv31)
		self.layers['conv31'] = self.conv31

		self.ReLU31 = nn.ReLU(True)
		self.features.append(self.ReLU31)
		self.layers['ReLU31'] = self.ReLU31

		self.conv32 = nn.Conv2d(
			in_channels=128,
			out_channels=128,
			kernel_size=3,
			padding=1
		)
		self.features.append(self.conv32)
		self.layers['conv32'] = self.conv32

		self.ReLU32 = nn.ReLU(True)
		self.features.append(self.ReLU32)
		self.layers['ReLU32'] = self.ReLU32

		self.pool3 = nn.MaxPool2d(2, 2)
		self.features.append(self.pool3)
		self.layers['pool3'] = self.pool3

		self.feature_dims = 4 * 4 * 128
		self.fc1 = nn.Linear(self.feature_dims, 512)
		self.classifier.append(self.fc1)
		self.layers['fc1'] = self.fc1

		self.fc1act = nn.Sigmoid()
		self.classifier.append(self.fc1act)
		self.layers['fc1act'] = self.fc1act

		self.fc2 = nn.Linear(512, n_class)
		self.classifier.append(self.fc2)
		self.layers['fc2'] = self.fc2

		self.initial_params = [param.data for param in self.parameters()]

	def getName(self):
		return "CifarNet"

	def forward(self, x, start=0, end=17):
		if start <= len(self.features) - 1:  # start in self.features
			for idx, layer in enumerate(self.features[start:]):
				x = layer(x)
				if idx == end:
					return x
			x = x.view(-1, self.feature_dims)
			for idx, layer in enumerate(self.classifier):
				x = layer(x)
				if idx + 15 == end:
					return x
			return x
		else:
			if start == 15:
				x = x.view(-1, self.feature_dims)
			for idx, layer in enumerate(self.classifier):
				if idx >= start - 15:
					x = layer(x)
				if idx + 15 == end:
					return x

	def get_params(self, end=17):
		params = []
		for layer in list(self.layers.values())[:end + 1]:
			params += list(layer.parameters())
		return params

	def restore_initial_params(self):
		for param, initial in zip(self.parameters(), self.initial_params):
			param.data = initial


def conv_block(in_channels, out_channels, pool=False):
	layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			  nn.Sequential(nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))]
	if pool: layers.append(nn.MaxPool2d(2))
	return nn.Sequential(*layers)
class ResNet9(nn.Module):
	def __init__(self, in_channels, num_classes,start=0, end=-1):
		super().__init__()

		self.conv1 = conv_block(in_channels, 64)
		self.conv2 = conv_block(64, 128, pool=True)
		self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
		self.conv3 = conv_block(128, 256, pool=True)
		self.conv4 = conv_block(256, 512, pool=True)
		self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

		self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
										nn.Flatten(),
										nn.Dropout(0.2),
										nn.Linear(512, num_classes))
		self.alllayers=nn.Sequential(self.conv1,self.conv2,self.res1,self.conv3,self.conv4,self.res2,self.classifier)
		self.totallayers=self.getTotalLayers(self.alllayers._modules.values())
		self.start = start
		self.setEnd(end)

	def getTotalLayers(self,layerlist):
		totallayers=0
		for l in layerlist:
			if isinstance(l,nn.Sequential):
				totallayers+=self.getTotalLayers(l._modules.values())
			elif not isinstance(l,(nn.ReLU,nn.Flatten)):
				totallayers+=1
		return totallayers

	def setEnd(self, end):
		if end == -1: end = self.totallayers
		self.end = end

	def setStartEnd(self, start, end=-1):
		# if start!=0: start+=1
		self.start = start
		self.setEnd(end)


	def getName(self):
		return "RestNet9"

	def forwardlayers(self,out,layerlist,layerindex = 0):
		for layer in layerlist:
			if isinstance(layer,nn.Sequential):
				out,layerindex=self.forwardlayers(out,layer._modules.values(),layerindex)
			elif not isinstance(layer,(nn.ReLU,nn.Flatten)):
				layerindex += 1
				if self.start<layerindex<=self.end:
					out=layer(out)
			else:
				if self.start<layerindex<=self.end:
					out=layer(out)

		return out,layerindex


	def forward(self, xb):
		# out=self.forwardlayers(xb,self.alllayers._modules.values())[0]
		out,layerindex = self.forwardlayers(xb,self.conv1._modules.values(),layerindex=0)
		out,layerindex = self.forwardlayers(out,self.conv2._modules.values(),layerindex=layerindex)
		out,layerindex = self.forwardlayers(out,self.res1._modules.values(),layerindex=layerindex)
		out,layerindex = self.forwardlayers(out,self.conv3._modules.values(),layerindex=layerindex)
		out,layerindex = self.forwardlayers(out,self.conv4._modules.values(),layerindex=layerindex)
		if layerindex>self.end:return out
		out = self.res2(out) + out
		out = self.classifier(out)
		return out

class AlexNet(nn.Module):
	def __init__(self,n_channels=3, num_classes=10, start=0, end=-1):
		# https://github.com/soapisnotfat/pytorch-cifar10/blob/master/models/AlexNet.py
		super().__init__()
		self.name="Alexnet"
		self.conv1=nn.Sequential(nn.Conv2d(n_channels,64,3,stride=2,padding=1),nn.ReLU(True))
		self.maxpool1=nn.MaxPool2d(2)
		self.conv2=nn.Sequential(nn.Conv2d(64,192,3,padding=1),nn.ReLU(True))
		self.maxpool2=nn.MaxPool2d(2)
		self.conv3=nn.Sequential(nn.Conv2d(192,384,3,padding=1),nn.ReLU(True))
		self.conv4=nn.Sequential(nn.Conv2d(384,256,3,padding=1),nn.ReLU(True))
		self.conv5=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU(True))
		self.maxpool3=nn.Sequential(nn.MaxPool2d(2),nn.Flatten())
		self.dropout1=nn.Dropout(p=0.5,inplace=False)
		self.linear1=nn.Sequential(nn.Linear(256 * 2 * 2,4096),nn.ReLU(True))
		self.dropout2=nn.Dropout(p=0.5,inplace=False)
		self.linear2=nn.Sequential(nn.Linear(4096,4096),nn.ReLU(True))
		self.linear3=nn.Linear(4096,num_classes)
		self.features=[self.conv1,self.maxpool1,self.conv2,self.maxpool2,self.conv3,self.conv4,self.conv5,self.maxpool3,
					  self.dropout1,self.linear1,self.dropout2,self.linear2,self.linear3]
		self.setStartEnd(start,end)

	def setEnd(self, end):
		if end == -1: end = len(self.features)
		self.end = end

	def setStartEnd(self, start, end=-1):
		# if start!=0: start+=1
		self.start = start
		self.setEnd(end)

	def getName(self):
		return self.name

	def forward(self, x):
		for feature in self.features[self.start:self.end]:
			x=feature(x)
		return x

class VGGSOTA(nn.Module):
	def __init__(self, start=0, end=-1,vgg_name="VGG"):
		super(VGGSOTA, self).__init__()
		model_cfg = {
			'VGG': [('C', 3, 32, 3, 32 * 32 * 32, 32 * 32 * 32 * 3 * 3 * 3), ('M', 32, 32, 2, 32 * 16 * 16, 0),
					('C', 32, 64, 3, 64 * 16 * 16, 64 * 16 * 16 * 3 * 3 * 32), ('M', 64, 64, 2, 64 * 8 * 8, 0),
					('C', 64, 64, 3, 64 * 8 * 8, 64 * 8 * 8 * 3 * 3 * 64),
					('D', 8 * 8 * 64, 128, 1, 64, 128 * 8 * 8 * 64),
					('D', 128, 10, 1, 10, 128 * 10)]
		}
		self.features, self.denses = self._make_layers(model_cfg[vgg_name])
		self.featuresele=nn.Sequential(*self.features)
		self.densesele=nn.Sequential(*self.denses)
		self.totallayers=len(self.features)+len(self.denses)
		self.setStartEnd(start,end)
		self._initialize_weights()

	def getName(self):
		return "VGGSOTA"

	def getTrainedParameter(self):
		layers= {}
		for i,layer in enumerate(self.features):
			if self.start<=i<self.end:
				for k,v in layer.state_dict().items():
					layers[f"featuresele.{i}.{k}"]=v
		executed=len(self.features)
		st,end=max(0,self.start-executed),self.end-executed
		for i,dense in enumerate(self.denses):
			if st<=i<end:
				for k,v in dense.state_dict().items():
					layers[f"densesele.{i}.{k}"]=v
		return layers


		# return {"features." + k: v for k, v in
		# 		nn.Sequential(*self.featurelayers[self.start:self.end]).state_dict().items()}
	def setEnd(self, end):
		if end == -1: end = self.totallayers + 1
		else: end+=1
		self.end = end

	def setStartEnd(self, start, end=-1):
		if start!=0: start+=1
		self.start = start
		self.setEnd(end)


	def forward(self, x):
		out = x
		for layer in self.features[self.start:self.end]:
			out = layer(out)
		if self.end <= len(self.features):
			return out
		out = out.view(out.size(0), -1)
		executed=len(self.features)
		for dense in self.denses[max(0,self.start-executed):self.end-executed]:
			out = dense(out)
		return out


	def _make_layers(self, cfg):
		features = []
		denses = []
		for x in cfg:
			in_channels, out_channels = x[1], x[2]
			kernel_size = x[3]
			if x[0] == 'M':
				features += [nn.MaxPool2d(kernel_size=kernel_size, stride=2)]
			if x[0] == 'D':
				denses += [nn.Linear(in_channels, out_channels)]
			if x[0] == 'C':
				features += [nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
							 nn.BatchNorm2d(out_channels),
							 nn.ReLU(inplace=True))]

		return features, denses

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)



VGGcfg = {
	'VGG8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 'M', 256, 'M', 256, 'M', 512, 'M', 512, 'M', 512, 'M', 512,
			  'M', 512, 'M', 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
	def __init__(self, vgg_name, n_channels=3, n_class=10, start=0, end=-1):
		super(VGG, self).__init__()
		self.n_channels = n_channels
		self.featurelayers = []
		self.start = start
		self.name = vgg_name
		self.features = self._make_layers(VGGcfg[vgg_name])
		self.setEnd(end)
		self.classifier = nn.Linear(512, n_class)

	def setEnd(self, end):
		if end == -1: end = len(self.features) + 1
		self.end = end

	def setStartEnd(self, start, end=-1):
		# if start!=0: start+=1
		self.start = start
		self.setEnd(end)

	def getName(self):
		return self.name

	def getTrainedParameter(self):
		return {"features." + k: v for k, v in
				nn.Sequential(*self.featurelayers[self.start:self.end]).state_dict().items()}

	def _make_layers(self, cfg):
		in_channels = self.n_channels
		for x in cfg:
			if x == 'M':
				self.featurelayers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				self.featurelayers += [nn.Sequential(nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
													 nn.ReLU(inplace=True))]
				in_channels = x
		self.featurelayers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*self.featurelayers)

	def forward(self, x):
		out = x
		for layer in self.featurelayers[self.start:self.end]:
			out = layer(out)
		if self.end <= len(self.featurelayers):
			return out
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out


VGGBNcfg = {
	'VGG11bn': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGBN(nn.Module):
	def __init__(self, vgg_name, n_channels=3, n_class=10, start=0, end=-1,pooling="AvgPool2d"):
		super(VGGBN, self).__init__()
		self.n_channels = n_channels
		self.featurelayers = []
		self.start = start
		self.name = vgg_name
		self.features = self._make_layers(VGGBNcfg[vgg_name],pooling)
		self.setEnd(end)
		if pooling == "AdaptiveAvgPool2d":
			self.classifier = nn.Linear(25088, n_class)
		else:
			self.classifier = nn.Linear(512, n_class)

	def setEnd(self, end):
		if end == -1: end = len(self.features) + 1
		self.end = end


	def setStartEnd(self, start, end=-1):
		# if start!=0: start+=1
		self.start = start
		self.setEnd(end)

	def getName(self):
		return self.name

	def getTrainedParameter(self):
		return {"features." + k: v for k, v in
				nn.Sequential(*self.featurelayers[self.start:self.end]).state_dict().items()}

	def _make_layers(self, cfg,pooling):
		in_channels = self.n_channels
		for x in cfg:
			if x == 'M':
				self.featurelayers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				self.featurelayers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
									   nn.Sequential(nn.BatchNorm2d(x),nn.ReLU(inplace=True))]
				in_channels = x
		if pooling=="AdaptiveAvgPool2d":
			self.featurelayers += [nn.AdaptiveAvgPool2d(output_size=(7, 7))]
		else:
			self.featurelayers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*self.featurelayers)

	def forward(self, x):
		try:
			out = x
			for layer in self.featurelayers[self.start:self.end]:
				out = layer(out)
			if self.end <= len(self.featurelayers):
				return out
			out = out.view(out.size(0), -1)
			out = self.classifier(out)
		except Exception as e:
			print(f"Expcpeiton in VGGBN {self.name} {e} try preTrain")
			raise e
		return out

# ['conv1.weight', 'bn1.0.weight', 'bn1.0.bias', 'bn1.0.running_mean', 'bn1.0.running_var', 'bn1.0.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.0.weight', 'layer1.0.bn1.0.bias', 'layer1.0.bn1.0.running_mean', 'layer1.0.bn1.0.running_var', 'layer1.0.bn1.0.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.0.weight', 'layer1.1.bn1.0.bias', 'layer1.1.bn1.0.running_mean', 'layer1.1.bn1.0.running_var', 'layer1.1.bn1.0.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.0.weight', 'layer1.2.bn1.0.bias', 'layer1.2.bn1.0.running_mean', 'layer1.2.bn1.0.running_var', 'layer1.2.bn1.0.num_batches_tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer2.0.conv1.weight', 'layer2.0.bn1.0.weight', 'layer2.0.bn1.0.bias', 'layer2.0.bn1.0.running_mean', 'layer2.0.bn1.0.running_var', 'layer2.0.bn1.0.num_batches_tracked', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight', 'layer2.1.bn1.0.weight', 'layer2.1.bn1.0.bias', 'layer2.1.bn1.0.running_mean', 'layer2.1.bn1.0.running_var', 'layer2.1.bn1.0.num_batches_tracked', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.0.weight', 'layer2.2.bn1.0.bias', 'layer2.2.bn1.0.running_mean', 'layer2.2.bn1.0.running_var', 'layer2.2.bn1.0.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.3.conv1.weight', 'layer2.3.bn1.0.weight', 'layer2.3.bn1.0.bias', 'layer2.3.bn1.0.running_mean', 'layer2.3.bn1.0.running_var', 'layer2.3.bn1.0.num_batches_tracked', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.0.weight', 'layer3.0.bn1.0.bias', 'layer3.0.bn1.0.running_mean', 'layer3.0.bn1.0.running_var', 'layer3.0.bn1.0.num_batches_tracked', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.0.weight', 'layer3.1.bn1.0.bias', 'layer3.1.bn1.0.running_mean', 'layer3.1.bn1.0.running_var', 'layer3.1.bn1.0.num_batches_tracked', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3.2.bn1.0.weight', 'layer3.2.bn1.0.bias', 'layer3.2.bn1.0.running_mean', 'layer3.2.bn1.0.running_var', 'layer3.2.bn1.0.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.0.weight', 'layer3.3.bn1.0.bias', 'layer3.3.bn1.0.running_mean', 'layer3.3.bn1.0.running_var', 'layer3.3.bn1.0.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var', 'layer3.3.bn2.num_batches_tracked', 'layer3.4.conv1.weight', 'layer3.4.bn1.0.weight', 'layer3.4.bn1.0.bias', 'layer3.4.bn1.0.running_mean', 'layer3.4.bn1.0.running_var', 'layer3.4.bn1.0.num_batches_tracked', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.4.bn2.num_batches_tracked', 'layer3.5.conv1.weight', 'layer3.5.bn1.0.weight', 'layer3.5.bn1.0.bias', 'layer3.5.bn1.0.running_mean', 'layer3.5.bn1.0.running_var', 'layer3.5.bn1.0.num_batches_tracked', 'layer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var', 'layer3.5.bn2.num_batches_tracked', 'layer4.0.conv1.weight', 'layer4.0.bn1.0.weight', 'layer4.0.bn1.0.bias', 'layer4.0.bn1.0.running_mean', 'layer4.0.bn1.0.running_var', 'layer4.0.bn1.0.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1.0.weight', 'layer4.1.bn1.0.bias', 'layer4.1.bn1.0.running_mean', 'layer4.1.bn1.0.running_var', 'layer4.1.bn1.0.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.0.weight', 'layer4.2.bn1.0.bias', 'layer4.2.bn1.0.running_mean', 'layer4.2.bn1.0.running_var', 'layer4.2.bn1.0.num_batches_tracked', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'fc.weight', 'fc.bias']
class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, in_planes, planes, stride=1,mid_plan=None):
		super(BasicBlock, self).__init__()
		self.features = []
		if mid_plan is None:
			self.featurescount = 4
			self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
			self.bn1 = nn.BatchNorm2d(planes)
			self.relu = nn.ReLU(True)
			self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn2 = nn.BatchNorm2d(planes)
			self.features.extend([self.conv1, self.bn1, self.conv2, self.bn2])
		else:
			self.featurescount = 6
			self.conv1 = nn.Conv2d(in_planes, mid_plan, kernel_size=1, stride=1, bias=False)
			self.bn1 = nn.BatchNorm2d(mid_plan)
			self.conv2 = nn.Conv2d(mid_plan, mid_plan, kernel_size=3, stride=stride, padding=1, bias=False)
			self.bn2 = nn.BatchNorm2d(mid_plan)
			self.conv3 = nn.Conv2d(mid_plan, planes, kernel_size=1, stride=1,  bias=False)
			self.bn3 = nn.BatchNorm2d(planes)
			self.relu = nn.ReLU(True)
			self.features.extend([self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3])

		self.downsample = None
		if stride != 1 and in_planes != self.expansion * planes:
			self.downsample = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)

	def forward(self, x, start=0, end=-1):
		if end == -1:
			end = self.featurescount
		out = x
		processed = 0
		lastlayer = None
		for feat in self.features:
			processed += 1
			if start < processed <= end:
				out = feat(out)
				lastlayer = feat
				if processed < end and isinstance(feat, nn.BatchNorm2d):
					out = self.relu(out)
					lastlayer = self.relu
			elif processed > end:
				if isinstance(lastlayer, nn.BatchNorm2d):
					out = self.relu(out)
				return out
		if self.downsample is not None:
			x = self.downsample(x)
		if out.shape == x.shape: out += x
		out = self.relu(out)
		# if end == len(self.features):
		# 	out = self.features[-1](out)
		return out

class ResNet(nn.Module):
	def __init__(self, version=18, n_channels=3, num_classes=10, start=0, end=-1,orignal=False,pooling="AvgPool2d"):
		super(ResNet, self).__init__()
		self.name = "ResNet" + str(version)
		Plans={m:[64,128,256,512] for m in [18,34]}
		Plans.update({10:[64,128,192,256],101:[(64,256),(128,512),(256,1024),(512,2048)]}) #https://gist.github.com/YugeTen/c2ffaaa2cfa0d9049335b4a8ee821b0d
		NumBlocks = {10:[1,1,1,1],18: [2, 2, 2, 2], 34: [3, 4, 6, 3],101:[3,4,23,3]}[version]
		Plans=Plans[version]
		self.in_planes = 64
		if orignal:
			self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
			self.bn1 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(True))
			self.maxpool  = nn.MaxPool2d(3,stride=2,padding=1)
			self.features=[(self.conv1,1),(self.bn1,1),(self.maxpool,1)]
			self.nonParameterLayerIndex=[3]
		else:
			self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn1 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(True))
			self.features = [(self.conv1, 1), (self.bn1, 1)]
			self.nonParameterLayerIndex=[]
		self.layer1 = self._make_layer(Plans[0], NumBlocks[0], stride=1)
		self.layer2 = self._make_layer(Plans[1], NumBlocks[1], stride=2)
		self.layer3 = self._make_layer(Plans[2], NumBlocks[2], stride=2)
		self.layer4 = self._make_layer(Plans[3], NumBlocks[3], stride=2)
		if pooling=="AdaptiveAvgPool2d" or version==101 or orignal:
			self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
		else:
			self.avgpool = nn.AvgPool2d(4)
		# self.fc = nn.Sequential(nn.Linear(512*BasicBlock.expansion, 1028),
		#                             nn.Linear(1028, 512),
		#                             nn.Linear(512*BasicBlock.expansion, num_classes))
		llinput=Plans[-1] if isinstance(Plans[-1],int) else Plans[-1][-1]
		self.fc = nn.Linear(llinput * BasicBlock.expansion, num_classes)
		self.features.extend([(self.avgpool, 1), (self.fc, 1)])
		self.start = start
		self.setEnd(end)

	def setEnd(self, end):
		if end == -1: end = sum([f[-1] for f in self.features]) + 1
		self.end = end

	def setStartEnd(self, start, end=-1):
		# if start!=0: start+=1
		self.start = start
		self.setEnd(end)

	def getName(self):
		return self.name

	def _make_layer(self, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		mid_plan,planes=(None,planes) if isinstance(planes, int) else planes

		for stride in strides:
			# print(self.in_planes, planes, stride)
			layers.append(BasicBlock(self.in_planes, planes, stride,mid_plan=mid_plan))
			self.in_planes = planes * BasicBlock.expansion
			self.features.append((layers[-1], layers[-1].featurescount))
		return nn.Sequential(*layers)

	def getTrainedParameter(self):
		def removeWeights(name):
			return ".".join(name.split(".")[:-1])

		uninqname = []
		for ln in [removeWeights(k) for k in self.state_dict().keys()]:
			if ln not in uninqname: uninqname.append(ln)
		for ind in self.nonParameterLayerIndex:
			uninqname.insert(ind-1,"NONPARAM")
		uninqname = uninqname[:self.end]
		keys = [k for k in self.state_dict().keys() if any([k.startswith(un) for un in uninqname])]
		return {k: self.state_dict()[k] for k in keys}


	def forward(self, x):
		out, processed = x, 0
		for feature, nlayer in self.features[:-1]:
			processed += nlayer
			if processed <= self.start: continue
			if nlayer in (0, 1):
				out = feature(out)
			else:
				startlocal, endlocal = 0, -1
				if processed - nlayer < self.start and self.start < processed:
					startlocal = self.start + nlayer - processed
				if processed > self.end:
					endlocal = self.end + nlayer - processed
					processed = self.end
				if startlocal == endlocal: return out
				out = feature(out, start=startlocal, end=endlocal)
			if processed >= self.end: return out
		out = out.view(out.size(0), -1)
		out = self.features[-1][0](out)
		return out



def getModels(modelname="VGG16bn", nchannel=3, sl=-1, nclasses=10, pretrained=False,orignal=False,pooling="AvgPool2d"):
	assert pooling in ("AvgPool2d","AdaptiveAvgPool2d")
	if modelname=="VGGSOTA":
		if pretrained: raise Exception("We don't have pretrained model for VGGSOTA")
		if sl == -1:
			model = VGGSOTA()
			return model
		model1, model2 = VGGSOTA(end=sl), VGGSOTA( start=sl)
		return model1, model2


	if modelname in ("VGG8","VGG16"):
		assert pooling=="AvgPool2d", pooling+" Not supported in VGG"
		if pretrained: raise Exception("We don't have pretrained model for VGG16")
		if sl == -1:
			model = VGG(modelname, n_channels=nchannel, n_class=nclasses)
			return model
		model1, model2 = VGG(modelname, n_channels=nchannel, n_class=nclasses, end=sl), VGGBN(modelname,
																							n_channels=nchannel,
																							n_class=nclasses, start=sl)
		return model1, model2

	elif modelname == "VGG16bn":
		if pretrained: orignal=True
		if orignal: pooling="AdaptiveAvgPool2d"

		modelpath = "Models/Pretrained/VGG16bn.pt"
		if pretrained and not os.path.exists(modelpath):
			modelpath = "../DataScientest/" + modelpath
		if sl == -1:
			model = VGGBN("VGG16bn", n_channels=nchannel, n_class=nclasses,pooling=pooling)
			if pretrained:
				print("Loading Imagenet Weights")
				model.load_state_dict(torch.load(modelpath), strict=False)
			return model
		model1, model2 = VGGBN("VGG16bn", n_channels=nchannel, n_class=nclasses, end=sl,pooling=pooling), VGGBN("VGG16bn",
													n_channels=nchannel,n_class=nclasses,start=sl,pooling=pooling)
		if pretrained:
			print("Loading Imagenet Weights for",modelname)
			weights = torch.load(modelpath)
			model1.load_state_dict(weights, strict=False)
			model2.load_state_dict(weights, strict=False)
		return model1, model2

	elif modelname=="ResNet9":
		if sl==-1:
			return ResNet9(in_channels=nchannel,num_classes=nclasses)
		else:
			return ResNet9(in_channels=nchannel,num_classes=nclasses,end=sl),ResNet9(in_channels=nchannel,num_classes=nclasses,start=sl)

	elif modelname.startswith("ResNet"):
		modelpath = "Models/Pretrained/Resnet"
		if pretrained and not os.path.exists(modelpath):
			modelpath = "../DataScientest/" + modelpath
		assert modelname.lower() in ["resnet10", "resnet18", "resnet34","resnet101"]
		if pretrained: orignal,pooling=True,"AdaptiveAvgPool2d"
		if sl == -1:
			# return ResNet18(n_channels=nchannel, num_classes=nclasses)
			# model=torch.hub.load('pytorch/vision:v0.10.0', modelname.lower(), pretrained=pretrained)
			# model.fc=nn.Sequential(torch.nn.Linear(model.fc.in_features,1028),torch.nn.Linear(1028,512),torch.nn.Linear(512,nclasses),)
			model = ResNet(int(modelname.lower().lstrip("resnet")), n_channels=nchannel, num_classes=nclasses,orignal=orignal,pooling=pooling)
			if pretrained:
				print("Loading Imagenet Weights For ", modelname)
				if orignal:
					print("Loading from ",modelpath + str(modelname.lower().lstrip("resnet")) + "_Orignal.pt")
					model.load_state_dict(torch.load(modelpath + str(modelname.lower().lstrip("resnet")) + "_Orignal.pt"),strict=False)
				else:
					print(modelpath + str(modelname.lower().lstrip("resnet")) + "2.pt")
					model.load_state_dict(torch.load(modelpath + str(modelname.lower().lstrip("resnet")) + "2.pt"))
			return model
		model1, model2 = (ResNet(int(modelname.lower().lstrip("resnet")), n_channels=nchannel, num_classes=nclasses, end=sl,orignal=orignal,pooling=pooling),
						  ResNet(int(modelname.lower().lstrip("resnet")),n_channels=nchannel,num_classes=nclasses,start=sl,orignal=orignal,pooling=pooling))
		if pretrained and orignal:
			print("Loading from ",modelpath + str(modelname.lower().lstrip("resnet")) + "_Orignal.pt")
			model1.load_state_dict(torch.load(modelpath + str(modelname.lower().lstrip("resnet")) + "_Orignal.pt"), strict=False)
			model2.load_state_dict(torch.load(modelpath + str(modelname.lower().lstrip("resnet")) + "_Orignal.pt"), strict=False)
		return model1, model2

	elif modelname == "alexnet":
		if sl == -1:
			# return AlexNet(n_channels=nchannel, num_classes=nclasses)
			AlexNet_Model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
			AlexNet_Model.classifier[1] = nn.Linear(9216, 4096)
			AlexNet_Model.classifier[4] = nn.Linear(4096, 1024)
			AlexNet_Model.classifier[6] = nn.Linear(1024, 10)
			return AlexNet_Model
		return AlexNet(n_channels=nchannel, num_classes=nclasses, end=sl), AlexNet(n_channels=nchannel, num_classes=nclasses,start=sl + 1)
	elif modelname == "efficientnetb0":
		# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=pretrained)
		model = torchvision.models.efficientnet_b0(pretrained=pretrained)
	elif "efficientnet_b" in modelname:
		model = torchvision.models.get_model(modelname)
	elif modelname == "VGG11":
		model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=pretrained)
	elif modelname == "VGG11bn":
		model = torchvision.models.vgg11_bn(pretrained=pretrained)
	elif modelname == "VGG13bn":
		model = torchvision.models.vgg13_bn(pretrained=pretrained)
	elif modelname == "VGG19bn":
		model = torchvision.models.vgg19_bn(pretrained=pretrained)
	elif modelname == "VGG19":
		model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=pretrained)
	# model.classifier[-1]=torch.nn.Linear(model.classifier[-1].in_features,nclasses)
	model.classifier[-1]=torch.nn.Sequential(torch.nn.Dropout(.2),torch.nn.Linear(model.classifier[-1].in_features,100),
				torch.nn.ReLU(inplace=True),torch.nn.Dropout(.2),torch.nn.Linear(100,nclasses))
	def getname():
		return modelname
	model.getName=getname
	return model
