import torch.nn as nn
import torch


class VGG(nn.Module):
    def __init__(self, num_classes=40, init_weights=False):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(23552, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)






















