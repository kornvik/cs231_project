import cv2
import numpy as np
import time
import json
import util


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


def initialize_kalman_filter(dt):
    kalman = cv2.KalmanFilter(6, 3)

    kalman.transitionMatrix = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=np.float32
    )

    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 3
    kalman.measurementNoiseCov = np.diag([1e-2, 1e-2, 1e-1]).astype(
        np.float32
    )  # Lower noise for x and y, higher for z
    kalman.errorCovPost = np.eye(6, dtype=np.float32) * 0.1

    return kalman


def mahalanobis_distance(innovation, innovation_covariance):
    return np.sqrt(innovation.T @ np.linalg.inv(innovation_covariance) @ innovation)


def main():
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

    # Initialize kalman filter
    delta_t = 1.0 / fps
    kalman = initialize_kalman_filter(delta_t)
    outlier_threshold = 25.0

    # Initialize mask for plotting point traces accross frames
    old_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)

    # Initialize Calibrated Camera Matrix
    K, dist_coeffs = initialize_camera_matrices()

    # Setup output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = create_video_writer(
        folder + file_name + "_plain_optical_flow_kalman.mp4",
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray_frame, gray_frame, old_points, None, **lk_params
        )

        if new_points is not None and status.any():
            good_new_points = new_points[status.ravel() == 1]
            good_old_points = old_points[status.ravel() == 1]

            for i, (new, old) in enumerate(zip(good_new_points, good_old_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(
                    mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 2
                )
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i], -1)

        success, rvec, tvec = cv2.solvePnP(
            object_points, good_new_points, K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
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

        innovation = tvec - kalman.measurementMatrix @ kalman.statePost
        innovation_covariance = (
            kalman.measurementMatrix @ kalman.errorCovPost @ kalman.measurementMatrix.T
            + kalman.measurementNoiseCov
        )
        mahalanobis_dist = mahalanobis_distance(innovation, innovation_covariance)
        if mahalanobis_dist < outlier_threshold:
            print("not outlier")
            kalman.correct(tvec)
            predicted = kalman.predict()

            old_gray_frame = gray_frame.copy()
            old_points = (
                good_new_points.reshape(-1, 1, 2)
                if good_new_points.size > 0
                else old_points
            )

            imgpts, _ = cv2.projectPoints(axis, rvec, predicted[:3], K, dist_coeffs)

            frame = util.draw(frame, old_points[1][0], imgpts)
            img = cv2.add(frame, mask)
            cv2.imshow("Frame", img)
            out.write(img)

        old_gray_frame = gray_frame.copy()
        old_points = (
            good_new_points.reshape(-1, 1, 2)
            if good_new_points.size > 0
            else old_points
        )

        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist_coeffs)

        frame = util.draw(frame, old_points[1][0], imgpts)
        img = cv2.add(frame, mask)
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
