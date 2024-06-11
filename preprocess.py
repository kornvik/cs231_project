import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

folder_name = "generated_images"


def project_points(points_3d, K, R, t):
    # Convert points to numpy array
    points_3d = np.array(points_3d, dtype=np.float32)

    # Project 3D points to 2D using the camera matrix, rotation matrix, and translation vector
    points_2d, _ = cv2.projectPoints(points_3d, R, t, K, distCoeffs=None)
    return points_2d


def euler_to_matrix(rotation):
    """Convert Euler angles to a rotation matrix."""
    x, y, z = rotation
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def plot_point(data, idx):

    # Extract camera parameters from the first frame
    first_frame = data["frames"][idx]
    camera_matrix = np.array(data["camera_matrix"])
    print("------------------")
    print(idx)
    print(camera_matrix)
    camera_location = np.array(first_frame["camera_location"])
    camera_rotation = np.array(first_frame["camera_rotation"])
    camera_rotation_rad = np.radians(camera_rotation)
    camera_orientation_matric = euler_to_matrix(camera_rotation_rad)
    rotation_matrix = camera_orientation_matric.T
    translation_vector = -1 * rotation_matrix @ camera_location

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    real_translation_vector = R @ translation_vector
    real_rotation = R @ rotation_matrix
    file_name = folder_name + first_frame["file_name"]
    image = cv2.imread(file_name)
    print("width hieght", image.shape)
    # print(image.size[1])
    if image is None:
        print(f"Error: Could not load image {file_name}")
        return

    # Define the 3D points to be projected
    points_3d = [
        [0.15, 0.15, 0],
        [-0.15, 0.15, 0],
        [-0.15, -0.15, 0],
        [0.15, -0.15, 0],
        [0, 0, 0],
    ]

    # Project the points onto the image
    points_2d = project_points(
        points_3d, camera_matrix, real_rotation, real_translation_vector
    )

    # Draw the projected points on the image
    for i, point in enumerate(points_2d):
        x, y = int(point[0][0]), int(point[0][1])
        print(x, y)
        if i == 0:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        elif i == 1:
            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
        elif i == 2:
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        else:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    # origin = points_2d[-1]
    # x, y = int(origin[0][0]), int(origin[0][1])
    # cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    output_path = f"projected_points_{idx}_check.png"  # Specify the output file path
    cv2.imwrite(output_path, image)


def get_vertices(data, idx):

    # Extract camera parameters from the first frame
    first_frame = data["frames"][idx]
    camera_matrix = np.array(data["camera_matrix"])
    camera_location = np.array(first_frame["camera_location"])
    camera_rotation = np.array(first_frame["camera_rotation"])
    camera_rotation_rad = np.radians(camera_rotation)
    camera_orientation_matric = euler_to_matrix(camera_rotation_rad)
    rotation_matrix = camera_orientation_matric.T
    translation_vector = -1 * rotation_matrix @ camera_location

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    real_translation_vector = R @ translation_vector
    real_rotation = R @ rotation_matrix

    # Define the 3D points to be projected
    points_3d = [
        [0.15, 0.15, 0],
        [-0.15, 0.15, 0],
        [-0.15, -0.15, 0],
        [0.15, -0.15, 0],
    ]

    # Project the points onto the image
    points_2d = project_points(
        points_3d, camera_matrix, real_rotation, real_translation_vector
    )

    # Draw the projected points on the image
    vertices = []
    for i, point in enumerate(points_2d):
        x, y = int(point[0][0]), int(point[0][1])
        vertices.append([x, y])

    return vertices


def read_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    return data


def main(json_file):
    data = read_json(json_file)
    # show_camera(data)
    print(len(data["frames"]))
    for i in range(len(data["frames"])):
        # plot_point(data, i)
        vertices = get_vertices(data, i)
        # print(vertices)
        data["frames"][i]["vertices"] = vertices
        # print(data["frames"][i])
    with open("result.json", "w") as fp:
        json.dump(data, fp)


if __name__ == "__main__":
    json_file = folder_name + "transforms.json"
    main(json_file)
