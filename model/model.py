import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import VGG19_Weights


class VGG19Backbone(nn.Module):
    def __init__(self):
        super(VGG19Backbone, self).__init__()
        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg19.features.children())[:24])

    def forward(self, x):
        return self.features(x)

# Implementation of DOPE model
class MultistageNetwork(nn.Module):
    def __init__(self, num_keypoints=4, num_stages=6):
        super(MultistageNetwork, self).__init__()
        self.backbone = VGG19Backbone()

        # Feature reduction layers
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # First stage layers
        self.stage1_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_keypoints, kernel_size=1, stride=1),
        )

        self.stages = nn.ModuleList()
        for _ in range(num_stages - 1):
            self.stages.append(self._make_stage(128 + num_keypoints, num_keypoints))

    def _make_stage(self, in_channels, num_keypoints):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, kernel_size=1, stride=1),
        )

    def forward(self, x):
        features = self.backbone(x)
        features = nn.ReLU(inplace=True)(self.conv1(features))
        features = nn.ReLU(inplace=True)(self.conv2(features))

        # First stage
        out1_2 = self.stage1_conv(features)

        # Store outputs for loss computation
        all_belief_maps = [out1_2]

        # Subsequent stages
        for stage in self.stages:
            combined_input = torch.cat([features, all_belief_maps[-1]], dim=1)
            stage_features = stage(combined_input)
            all_belief_maps.append(stage_features)

        return all_belief_maps

def load_model(model_path="belief_map_model.pth", device="cuda"):

    # Load the saved state dictionaries
    checkpoint = torch.load(model_path, map_location=device)
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]

    model = MultistageNetwork(num_keypoints=4, num_stages=6)
    model.load_state_dict(checkpoint["model_dict"])

    model.to(device)
    print(f"Model loaded from {model_path}, epoch: {epoch}, loss: {loss:.4f}")

    return model, epoch, loss, train_losses, val_losses


# Test the network with a dummy input
if __name__ == "__main__":
    model = MultistageNetwork(num_keypoints=4, num_stages=6)
    dummy_input = torch.randn(1, 3, 400, 400)
    outputs = model(dummy_input)
    print(len(outputs))
    for output in outputs:
        print(output.shape)
