from data import PoseDatasetWithBeliefMaps
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
import new_data

from model import MultistageNetwork
import training
import testing
import model

json_file = "result.json"
root_dir = "../generated_image_no_back_gamma_xyz"

# Hyperparameters
num_epochs = 5
batch_size = 16
learning_rate = 0.0001

voc_root = "backgrounds"
background_images = new_data.load_stanford_backgrounds(voc_root)
print(len(background_images))

# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((400, 400)),
        new_data.RandomBlackoutAroundCenter(),
        new_data.RandomBackground(background_images, resize=(400, 400)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(2.0, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

gen = torch.Generator()
gen.manual_seed(0)

dataset = PoseDatasetWithBeliefMaps(json_file, root_dir, transform=transform, pct=0.1)

# dataset.show_batch(n=3, isMainRun=False)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=gen
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

trained_model, epoch, loss, train_losses, val_losses = model.load_model()
testing.evaluate(trained_model, test_loader)
testing.plot_predictions(test_dataset, trained_model, num_images=1)
testing.test_with_arbitrary_image(trained_model, "test_with_bg_2.png")
