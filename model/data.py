import json
import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

random.seed(10)


def load_stanford_backgrounds(dataset_dir):
    background_images = []
    images_dir = os.path.join(dataset_dir, "images")
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(images_dir, filename)
            background_images.append(Image.open(img_path).convert("RGB"))
    return background_images


class RandomBlackoutAroundCenter(object):
    def __init__(self, blackout_size=(30, 30), num_patches=1):
        self.blackout_size = blackout_size
        self.num_patches = num_patches

    def __call__(self, img):
        img = transforms.ToTensor()(img)  # Convert PIL image to tensor
        _, h, w = img.shape
        center_h, center_w = h // 2, w // 2

        for _ in range(self.num_patches):
            blackout_h, blackout_w = self.blackout_size
            top_offset = random.randint(
                -h // 4, h // 4
            )  # Random offset around the center
            left_offset = random.randint(-w // 4, w // 4)
            top = max(0, center_h + top_offset - blackout_h // 2)
            left = max(0, center_w + left_offset - blackout_w // 2)
            bottom = min(h, top + blackout_h)
            right = min(w, left + blackout_w)

            img[:, top:bottom, left:right] = 0

        img = transforms.ToPILImage()(img)  # Convert back to PIL image
        return img


class RandomBackground:
    def __init__(self, background_images, resize=(224, 224)):
        self.background_images = background_images
        self.resize = resize

    def __call__(self, img):
        background_img = random.choice(self.background_images).resize(self.resize)
        # Resize the input image

        # Convert both images to numpy arrays
        img_np = np.array(img)

        background_img = background_img.resize(self.resize)
        background_np = np.array(background_img)
        # print(background_np.shape)

        # Create a mask that considers all pixels as non-transparent
        mask = np.all(img_np != 0, axis=-1)

        # Use vectorized operations to replace the background
        combined_np = background_np.copy()
        combined_np[mask] = img_np[mask]

        # Convert back to PIL image
        combined_img = Image.fromarray(combined_np)
        return combined_img


class RandomAffineTransform:
    def __init__(self, degrees, translate, scale):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        # self.shear = shear

    def __call__(self, img, keypoints):
        # Apply random affine transformation
        angle = random.uniform(-self.degrees, self.degrees)
        # angle = 0
        max_dx = self.translate[0] * img.size[0]
        max_dy = self.translate[1] * img.size[1]
        translations = [
            int(random.uniform(-max_dx, max_dx)),
            int(random.uniform(-max_dy, max_dy)),
        ]
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        # shear_x = random.uniform(-self.shear[0], self.shear[0])
        # shear_y = random.uniform(-self.shear[1], self.shear[1])

        img = F.affine(
            img,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )

        # Transform keypoints
        keypoints = self.transform_keypoints(
            keypoints, angle, translations, scale, img.size
        )
        return img, keypoints

    def transform_keypoints(self, keypoints, angle, translations, scale, image_size):
        # Convert keypoints to homogeneous coordinates
        keypoints_h = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))

        # Calculate affine transformation matrix
        center = (image_size[0] / 2, image_size[1] / 2)
        angle = np.deg2rad(angle)
        # Transformation matrix
        T = np.array([[1, 0, translations[0]], [0, 1, translations[1]], [0, 0, 1]])
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

        # Center translation matrices
        C = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
        C_inv = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])

        # Combine transformations
        M = C_inv @ T @ R @ S @ C

        # Transform keypoints
        transformed_keypoints_h = keypoints_h @ M.T
        transformed_keypoints = (
            transformed_keypoints_h[:, :2] / transformed_keypoints_h[:, 2:]
        )

        return transformed_keypoints


class PoseDatasetWithBeliefMaps(Dataset):

    def __init__(
        self, json_file, root_dir, transform=None, affine_transform=None, pct=1.0
    ):
        with open(json_file) as f:
            self.data = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        self.affine_transform = affine_transform
        self.dataset_size = int(len(self.data["frames"]) * pct)
        sample_img_path = os.path.join(root_dir, self.data["frames"][0]["file_name"])
        sample_img = Image.open(sample_img_path)
        self.image_size = sample_img.size
        print("Image size", self.image_size)

    def __len__(self):
        return self.dataset_size

    def get_vertices(self, idx):
        frame = self.data["frames"][idx]
        keypoints = np.array(frame["vertices"], dtype=np.float32)
        return torch.tensor(keypoints, dtype=torch.float32)

    def get_plain_image(self, idx):
        frame = self.data["frames"][idx]
        img_path = os.path.join(self.root_dir, frame["file_name"])
        image = Image.open(img_path).convert("RGB")
        return image

    def generate_belief_maps(self, image_size, keypoints, sigma=2):
        w, h = image_size
        belief_maps = np.zeros(shape=(len(keypoints), int(h), int(w)), dtype=np.float32)
        for i, (x, y) in enumerate(keypoints):
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            belief_maps[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        return belief_maps

    def __getitem__(self, idx):
        frame = self.data["frames"][idx]
        img_path = os.path.join(self.root_dir, frame["file_name"])
        image = Image.open(img_path).convert("RGB")

        keypoints = np.array(frame["vertices"], dtype=np.float32)

        if self.affine_transform:
            image, keypoints = self.affine_transform(image, keypoints)

        if self.transform:
            image = self.transform(image)

        _, h, w = image.shape  # type: ignore

        # print("Transformed image size", image.shape)
        # Normalize keypoints to range [0, 1]
        keypoints[:, 0] /= self.image_size[0]  # /1280
        keypoints[:, 0] *= w / 8  # *400/8 = * 50
        keypoints[:, 1] /= self.image_size[1]  # /720
        keypoints[:, 1] *= h / 8

        # Generate belief maps
        belief_maps = self.generate_belief_maps((w / 8, h / 8), keypoints)

        return image, torch.tensor(belief_maps, dtype=torch.float32), keypoints

    def denormalize(self, img):
        # Assuming the transformation used standard normalization values
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        img = img * std + mean
        return img

    def tensor_to_image(self, tensor):
        img = tensor.permute(1, 2, 0).cpu().numpy()  # Change shape to (H, W, C)
        img = np.clip(img * 255, 0, 255).astype(
            np.uint8
        )  # Denormalize and convert to uint8
        return Image.fromarray(img)

    def show_batch(self, n=3, key_points=4, isMainRun=True):
        fig, axs = plt.subplots(n, key_points + 1, figsize=(15, 15))
        fig.tight_layout()

        for i in range(n):
            print("plotting", i)
            rand_idx = random.randint(0, len(self) - 1)
            img, belief_maps, _ = self.__getitem__(rand_idx)
            for bl in belief_maps:
                print(bl.shape)

            img = self.denormalize(img)

            if isinstance(img, torch.Tensor):
                img = self.tensor_to_image(img)

            # Display the image
            axs[i, 0].imshow(img)
            axs[i, 0].set_title("Image")
            axs[i, 0].axis("off")

            # Show each heatmap separately
            belief_maps_np = belief_maps.numpy()
            for k in range(belief_maps_np.shape[0]):
                heatmap = belief_maps_np[k]
                heatmap = (heatmap - heatmap.min()) / (
                    heatmap.max() - heatmap.min()
                )  # Normalize heatmap to [0, 1]
                axs[i, k + 1].imshow(heatmap, cmap="jet")
                axs[i, k + 1].set_title(f"Heatmap {k + 1}")
                axs[i, k + 1].axis("off")
                print(k + 1)
                # axs[i, 0].imshow(
                #     heatmap, cmap="jet", alpha=0.25
                # )  # Overlay heatmap with colormap
                # Overlay keypoints on the image
                max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                axs[i, 0].imshow(
                    heatmap, cmap="jet", alpha=0.25
                )  # Overlay heatmap with colormap
                axs[i, 0].scatter(
                    max_idx[1] * 8, max_idx[0] * 8, c="r", s=40, marker="x"
                )
        if isMainRun:
            plt.savefig("batch_image_belief.png")
        else:
            plt.savefig("batch_image_belief_test.png")
        plt.show()
        plt.close(fig)  # Close the figure to avoid memory issues


if __name__ == "__main__":
    json_file = "result.json"
    root_dir = "generated_images"

    stanford_bg = "backgrounds"
    background_images = load_stanford_backgrounds(stanford_bg)
    print(len(background_images))
    transform = transforms.Compose(
        [
            transforms.Resize((400, 400)),
            RandomBlackoutAroundCenter(),
            RandomBackground(background_images, resize=(400, 400)),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.2, saturation=0.2, hue=0.3
            ),
            transforms.GaussianBlur(kernel_size=5, sigma=(2.0, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    affine_transform = RandomAffineTransform(
        degrees=30, translate=(0.2, 0.2), scale=0.4
    )

    dataset = PoseDatasetWithBeliefMaps(
        json_file, root_dir, transform=transform, affine_transform=affine_transform
    )

    # Show a batch of images
    dataset.show_batch(n=3)
