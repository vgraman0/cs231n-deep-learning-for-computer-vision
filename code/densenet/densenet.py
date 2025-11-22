import torch
import torch.nn as nn
import torch.nn.functional as F

class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        concat_features = torch.cat(inputs, 1)
        return self.conv1(self.relu1(self.norm1(concat_features)))
    
    def forward(self, inputs):
        bottleneck_output = self.bn_function(inputs)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=True)

        return new_features

class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, dim=1)

class DenseNet(nn.Module):
    """
    Generic DenseNet, configured for CIFAR-10 style inputs (3x32x32).
    """

    def __init__(
        self,
        growth_rate=12,
        block_config=(16, 16, 16),
        num_init_features=24,
        bn_size=4,
        drop_rate=0.0,
        num_classes=10,
    ):
        super().__init__()

        # CIFAR-10 stem: 3x3 conv, stride 1
        self.stem = nn.Sequential(
            nn.Conv2d(
                3, num_init_features,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        num_features = num_init_features

        # Dense blocks + transition layers
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            # Add transition layer between DenseBlocks
            if i != len(block_config) - 1:
                trans = Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.transitions.append(trans)
                num_features = num_features // 2

        # Final norm
        self.norm_final = nn.BatchNorm2d(num_features)

        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)

        # (Optional) init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.norm_final(x)
        x = F.relu(x, inplace=True)

        # Global average pooling over H, W -> 1x1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x

