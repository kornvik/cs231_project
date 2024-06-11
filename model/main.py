from data import PoseDatasetWithBeliefMaps
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
import new_data

from model import MultistageNetwork
import training
import testing

json_file = "result.json"
root_dir = "generated_images"


# Hyperparameters
num_epochs = 25
batch_size = 16
learning_rate = 0.0001

stanford_bg_root = "backgrounds"
background_images = new_data.load_stanford_backgrounds(stanford_bg_root)
print(len(background_images))

# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((400, 400)),
        new_data.RandomBlackoutAroundCenter(),
        new_data.RandomBackground(background_images, resize=(400, 400)),
        transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.3),
        transforms.GaussianBlur(kernel_size=5, sigma=(2.0, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

affine_transform = new_data.RandomAffineTransform(
    degrees=30, translate=(0.2, 0.2), scale=0.4
)

gen = torch.Generator()
gen.manual_seed(0)

dataset = PoseDatasetWithBeliefMaps(json_file, root_dir, transform=transform, pct=1.0)
dataset.show_batch(n=3)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=gen
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MultistageNetwork(num_keypoints=4).cuda()

# Train the model
trained_model = training.train(
    model,
    train_loader,
    val_loader,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    model_path="map_model.pth",
)
testing.evaluate(model, test_loader)
testing.plot_predictions(test_dataset, model, num_images=10)
testing.test_with_arbitrary_image(trained_model, "test_with_bg_3.png")
