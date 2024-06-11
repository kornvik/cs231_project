import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import model
from scipy.ndimage import gaussian_filter
import testing


class PoseDeepEstimator:
    def __init__(self, model_path="belief_map_model.pth", device=None):
        self.idx = 0
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model, _, _, _, _ = model.load_model(model_path, self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def predict(self, image):
        # plt.imshow(image)
        transform = transforms.Compose(
            [
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)

        self.model.eval()

        with torch.no_grad():
            belief_maps = self.model(image)

        final_output = belief_maps[-1].squeeze().cpu().numpy()
        key_points = testing.extract_keypoint(final_output)
        print(key_points)

        normalized_keypoints = []
        colors = ["red", "green", "blue", "white"]

        for i, keypoint in enumerate(key_points):
            x = int(keypoint[0] * 1280 / 400)
            y = int(keypoint[1] * 720 / 400)
            normalized_keypoints.append([x, y])
            print(x, y)
            # plt.scatter(x, y, c=colors[i % len(colors)], marker="x")
        # plt.show()

        return np.array(normalized_keypoints)


# Example usage
if __name__ == "__main__":
    image_path = "test_bg.png"  # Replace with your image path
    model_path = "belief_map_model_11.pth"  # Replace with your model path
    pose_estimator = PoseDeepEstimator(model_path)

    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    normalized_keypoints = pose_estimator.predict(image)
    print(normalized_keypoints)
