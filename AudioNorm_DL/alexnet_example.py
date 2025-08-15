# AlexNet-style architecture for audio normalization
class AlexNetAudio(nn.Module):
    def __init__(self, n_mels=N_MELS, additional_features_dim=9):
        super().__init__()
        
        # AlexNet-style CNN layers
        self.features = nn.Sequential(
            # Conv1: Large filters like AlexNet
            nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: Group convolution (AlexNet style)
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1, groups=2),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # AlexNet-style classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6 + additional_features_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1)  # Output: gain in dB
        )
        
    def forward(self, spectrogram, additional_features):
        # Add channel dimension: (batch, 1, n_mels, time_frames)
        x = spectrogram.unsqueeze(1)
        
        # Extract features using AlexNet-style conv layers
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Concatenate with additional audio features
        x = torch.cat([x, additional_features], dim=1)
        
        # Classify (predict gain)
        x = self.classifier(x)
        return x
