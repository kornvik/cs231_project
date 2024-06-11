import cv2
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
from model.model_load import PoseDeepEstimator
import util


checkpoint = "./extract_pile/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
device = "mps"
sam.to(device=device)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def remove_background(frame, showFirstFrame=False):
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor = SamPredictor(sam)
    predictor.set_image(color_frame)
    h, w, _ = color_frame.shape

    # you may need to adjust this to segment the pile properly
    input_point = np.array(
        [
            [w / 2, h / 2 - 50],
            [w / 2 + 120, h / 2 - 50],
            [w / 2, h / 2 + 100],
            [w / 2 + 50, h / 2 + 150],
        ]
    )
    input_label = np.array([1, 1, 1, 1])
    input_box = np.array([480, 250, 750, 700])
    masks, _, _ = predictor.predict(
        multimask_output=False,
        point_coords=input_point,
        point_labels=input_label,
        # box=input_box[None, :],
    )
    segmented_image = cv2.bitwise_and(
        color_frame, color_frame, mask=masks[0].astype(np.uint8)
    )
    if showFirstFrame:
        plt.figure(figsize=(10, 10))
        plt.imshow(color_frame)
        show_mask(masks[0], plt.gca())
        show_points(input_point, input_label, plt.gca())
        show_box(input_box, plt.gca())
        plt.axis("off")
        plt.show()
    return cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)


def initialize_video_capture(file_path):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Unable to read video file.")
    return cap, frame, fps


def create_video_writer(file_path, frame_width, frame_height, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height))


def load_feature_points(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        points = np.array(data["pre_identified_feature_points"], dtype=np.float32)
    return points[:3] if len(points) > 3 else points


def initialize_camera_matrices():
    K = np.array(
        [
            [897.28329969, 0.0, 639.20530177],
            [0.0, 901.30693926, 352.37955671],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.array(
        [
            -1.03876805e-02,
            9.11113581e-01,
            -6.08293510e-03,
            -2.88580271e-04,
            -3.31718733e00,
        ]
    )
    return K, dist_coeffs


def main():
    # More details in https://github.com/opencv/opencv/issues/8813#issuecomment-390462446
    resolveMirrorAmbiguity = True

    # Axis for plotting pile axis
    axis = np.float32([[30, 0, 0], [0, 30, 0], [0, 0, 30]]).reshape(-1, 3)

    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]

    # Scene config
    folder = "video/scene4/"
    file_name = "video"
    full_path = folder + file_name

    # Optical flow param
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    cap, frame, fps = initialize_video_capture(full_path + ".mp4")

    # Remove background
    background_removed_frame = remove_background(frame, True)
    old_gray_frame = cv2.cvtColor(background_removed_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(frame)

    # Initialize Calibrated Camera Matrix
    K, dist_coeffs = initialize_camera_matrices()

    # Setup output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = create_video_writer(
        folder + file_name + "_optical_flow_segment.mp4",
        frame_width,
        frame_height,
        fps,
    )

    # Setup plot while code is running
    cv2.namedWindow("Frame")

    # Load preidentified feature points
    old_points = load_feature_points(folder + "pre_identified_feature_points.json")
    object_points = np.array([[30, 0, 0], [0, 0, 0], [30, 0, 30]], dtype=np.float32)

    initialVec = None
    lastVec = None
    initialRVec = None
    lastRVec = None
    last_frame = None

    # Load model
    poseEstimate = PoseDeepEstimator(model_path="belief_map_model_11.pth")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        background_removed_frame = remove_background(frame)
        gray_frame = cv2.cvtColor(background_removed_frame, cv2.COLOR_BGR2GRAY)

        new_points = poseEstimate.predict(
            cv2.cvtColor(background_removed_frame, cv2.COLOR_BGR2RGB)
        )
        prev_points = None

        if new_points is not None:
            if prev_points is not None and np.all(prev_points):
                for i, new_point in enumerate(new_points):
                    if new_points[i][0] == 0 and new_points[i][1] == 0:
                        new_points[i] = prev_points[i]
                prev_tracked_points = prev_points
            else:
                prev_tracked_points = new_tracked_points
            new_tracked_points = np.array(new_points, dtype=np.float32)
            new_tracked_points = np.delete(new_tracked_points, 2, axis=0)
    
            for i, (new, old) in enumerate(zip(new_tracked_points, prev_tracked_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = (
                    int(a),
                    int(b),
                    int(c),
                    int(d),
                )
                if a != 0 and b != 0:
                    frame = cv2.circle(frame, (a, b), 5, util.get_color(i), -1)

        success, rvec, tvec = cv2.solvePnP(
            object_points, new_tracked_points, K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
        )
        R, _ = cv2.Rodrigues(rvec)

        if resolveMirrorAmbiguity and np.cross(R.T[:, 0], R.T[:, 1])[2] < 0:
            R *= np.array([[1, -1, 1], [1, -1, 1], [-1, 1, -1]])
            rvec, _ = cv2.Rodrigues(R)

        if initialVec is None:
            initialVec = tvec
            initialRVec = rvec
        lastVec = tvec
        lastRVec = rvec

        R_initial, _ = cv2.Rodrigues(initialRVec)

        depth_text = f"Depth: {(R_initial.T @ (lastVec - initialVec))[1]} cm."

        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, K, dist_coeffs)

         # Now update the previous frame and previous points for next loop optical flow
        prev_points = new_tracked_points

        frame = util.draw(frame, prev_points[2], imgpts)
        img = cv2.add(frame, mask)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20, 30)
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 3
        cv2.putText(
            img,
            depth_text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
        cv2.imshow("Frame", img)
        out.write(img)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("initialVec", initialVec)
    print("lastVec", lastVec)
    print("initialRVec", np.degrees(initialRVec))
    print("lastRVec", np.degrees(lastRVec))

    R_initial, _ = cv2.Rodrigues(initialRVec)
    if initialRVec[1] < 0:
        rvec *= -1
        R_initial, _ = cv2.Rodrigues(rvec)

    print(lastVec - initialVec, "delta tvec")
    print((R_initial.T @ (lastVec - initialVec))[1], "pile depth, should be positive")


if __name__ == "__main__":
    main()
