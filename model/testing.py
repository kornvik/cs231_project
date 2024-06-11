import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import tqdm
import random
import cv2
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

# From https://github.com/NVlabs/Deep_Object_Pose
def find_local_peaks(belief_map, sigma=15, threshold=0.1):
    map = gaussian_filter(belief_map, sigma=sigma).astype(np.float64)
    p = 1
    map_left = np.zeros(map.shape, dtype=np.float64)
    map_left[p:, :] = map[:-p, :]
    map_right = np.zeros(map.shape, dtype=np.float64)
    map_right[:-p, :] = map[p:, :]
    map_up = np.zeros(map.shape, dtype=np.float64)
    map_up[:, p:] = map[:, :-p]
    map_down = np.zeros(map.shape, dtype=np.float64)
    map_down[:, :-p] = map[:, p:]

    peaks_binary = np.logical_and.reduce(
        (
            map >= map_left,
            map >= map_right,
            map >= map_up,
            map >= map_down,
            map > threshold,
        )
    )
    peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])
    peaks = list(peaks)
    print("peaks", peaks)

    if not peaks:
        return []  # No peaks found above threshold

    # Compute the weighted average for localizing the peaks
    win = 5
    ran = win // 2  # 2
    max_weighted_value = -np.inf
    for p_value in range(len(peaks)):
        p = peaks[p_value]  # (x,y) of the peak
        weights = np.zeros((win, win))
        i_values = np.zeros((win, win))
        j_values = np.zeros((win, win))
        for i in range(-ran, ran + 1):  # -2,-1,0,1,2
            for j in range(-ran, ran + 1):  # -2,-1,0,1,2
                if (
                    p[1] + i < 0
                    or p[1] + i >= map.shape[0]
                    or p[0] + j < 0
                    or p[0] + j >= map.shape[1]
                ):
                    continue
                i_values[j + ran, i + ran] = p[1] + i
                j_values[j + ran, i + ran] = p[0] + j
                weights[j + ran, i + ran] = map[p[1] + i, p[0] + j]
        OFFSET_DUE_TO_UPSAMPLING = 0.4395
        try:
            weighted_avg = (
                np.average(j_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING,
                np.average(i_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING,
            )
            weighted_value = map[p[1], p[0]]
        except:
            weighted_avg = (
                p[0] + OFFSET_DUE_TO_UPSAMPLING,
                p[1] + OFFSET_DUE_TO_UPSAMPLING,
            )
            weighted_value = map[p[1], p[0]]

        if weighted_value > max_weighted_value:
            max_weighted_value = weighted_value
            best_peak = weighted_avg

    return [best_peak] if max_weighted_value > threshold else []
    # return peaks_avg


def evaluate(model, test_loader, device="cuda"):
    test_loss = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for images, belief_maps, _ in tqdm.tqdm(
            test_loader, desc="Testing", unit="batch"
        ):
            images, belief_maps = images.to(device), belief_maps.to(device)
            outputs = model(images)
            loss = torch.tensor(0).float().cuda()
            for stage in range(len(outputs)):
                loss += (
                    (outputs[stage] - belief_maps) * (outputs[stage] - belief_maps)
                ).mean()

            # loss = belief_map_loss(outputs, belief_maps)
            test_loss += loss.item()
            num_batches += 1

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")
    return test_loss


def extract_keypoint(belief_maps):
    predicted_keypoints = []

    for j in range(belief_maps.shape[0]):
        # print("idx", j)
        belief_map = belief_maps[j]
        peaks = find_local_peaks(belief_map, sigma=4, threshold=0.1)
        # print("len_peak", len(peaks))
        if len(peaks) == 0:
            predicted_keypoints.append([0, 0])
        for peak in peaks:
            # print("peak", peak)
            x, y = peak
            predicted_keypoints.append([x * 8, y * 8])

    return np.array(predicted_keypoints)


def plot_predictions(subset, model, num_images=5, device="cuda"):
    colors_gt = [(0, 0, 100), (0, 100, 0), (100, 0, 0), (100, 100, 100)]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
    model.eval()
    dataset = subset.dataset
    random.seed(10)
    for i in range(num_images):
        idx = random.randint(0, len(dataset) - 1)
        print("idx", idx)
        plain_image = dataset.get_plain_image(idx)
        plain_image_np = np.array(plain_image)

        image, belief_maps, vertices = dataset[idx]

        renderable_image = dataset.denormalize(image)
        renderable_image = dataset.tensor_to_image(renderable_image)
        renderable_image = np.array(renderable_image)
        # print("shape", renderable_image.shape)

        print(vertices)
        # Get the model's prediction

        with torch.no_grad():
            outputs = model(image.unsqueeze(0).to(device))
            final_output = outputs[-1].squeeze().cpu().numpy()

            # Extract predicted keypoints from the belief maps
            predicted_keypoints = extract_keypoint(final_output)

            vertices[:, 0] *= 8
            vertices[:, 1] *= 8

            print("details")
            print("pred_position", predicted_keypoints)
            print("true_position", vertices)

            for j, point in enumerate(predicted_keypoints):
                x, y = int(point[0]), int(point[1])
                cv2.circle(renderable_image, (x, y), 5, colors[j], -1)
            for j, point in enumerate(vertices):
                x, y = int(point[0]), int(point[1])
                cv2.circle(renderable_image, (x, y), 5, colors_gt[j], -1)
            cv2.imwrite(f"projected_image_{i}_belief.png", renderable_image)

            print(f"Prediction saved to projected_image_{i}_belief.png")


def preprocess_image(image, transform):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def predict(model, image, device):
    transform = transforms.Compose(
        [
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = preprocess_image(image, transform)
    image = image.to(device)

    model.eval()

    with torch.no_grad():
        belief_maps = model(image)

    final_output = belief_maps[-1].squeeze().cpu().numpy()
    key_points = extract_keypoint(final_output)

    return key_points


def visualize_keypoints(image, keypoints):
    plt.imshow(image)
    colors = ["red", "green", "blue", "white"]  # Matplotlib color names
    for i, keypoint in enumerate(keypoints):
        x = int(keypoint[0] * image.shape[1] / 400)
        y = int(keypoint[1] * image.shape[0] / 400)
        plt.scatter(x, y, c=colors[i % len(colors)], marker="x")
    plt.savefig("test_real_image_belief.png")
    plt.show()


def test_with_arbitrary_image(model, image_path, device="cuda"):
    # Load the arbitrary image
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    # Predict the keypoints
    keypoints = predict(model, image, device)
    print(keypoints)

    # Visualize the result
    visualize_keypoints(image, keypoints)
