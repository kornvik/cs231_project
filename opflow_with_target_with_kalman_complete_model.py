import cv2
import numpy as np
import time
import json
import util


class Custom_EKF:
    def __init__(
        self,
        dt=0.1,
        m=3000,
        k=100.0,
        c=10.0,
        cr=5.0,
        Q=np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # Process noise covariance
        R=np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # Measurement noise covariance
        P=np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # Initial estimate covariance
        x0=np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),  # Initial state for x, y, z positions and velocities
        F0=100.0,
        T=1.0,
        outlier_threshold=10,  # Mahalanobis distance threshold for outlier detection
    ):
        self.dt = dt
        self.m = m  # Mass of the pile
        self.k = k  # Stiffness of the soil
        self.c = c  # Damping coefficient of the pile
        self.cr = cr  # Damping coefficient of the soil
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Initial estimate covariance
        self.x = x0  # Initial state
        self.F0 = F0  # Impulse force magnitude
        self.T = T  # Period of the hammer force
        self.t = 0  # Initial time
        self.outlier_threshold = (
            outlier_threshold  # Mahalanobis distance threshold for outlier detection
        )

    def f(self, x, u):
        # State transition model
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + x[3] * self.dt
        x_next[1] = x[1] + x[4] * self.dt
        x_next[2] = x[2] + x[5] * self.dt
        x_next[3] = x[3]
        x_next[4] = x[4]
        x_next[5] = x[5] + self.dt / self.m * (
            u - (self.c + self.cr) * x[5] - self.k * x[2]
        )
        return x_next

    def h(self, x):
        # Measurement model
        return x

    def jacobian_F(self, x):
        # Jacobian of the state transition model
        F = np.array(
            [
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [
                    0,
                    0,
                    -self.dt * self.k / self.m,
                    0,
                    0,
                    1 - self.dt * (self.c + self.cr) / self.m,
                ],
            ]
        )
        return F

    def jacobian_H(self, x):
        # Jacobian of the measurement model
        H = np.eye(6)
        return H

    def predict(self):
        # Predict the next state
        if np.isclose(self.t % self.T, 0, atol=self.dt):
            u = self.F0  # Apply the impulse force
        else:
            u = 0  # No force

        self.x = self.f(self.x, u)
        F = self.jacobian_F(self.x)
        self.P = F @ self.P @ F.T + self.Q
        self.t += self.dt  # Increment time

    def update(self, z):
        # Update the state with the measurement z
        H = self.jacobian_H(self.x)
        y = z - self.h(self.x)  # Innovation
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain

        # Calculate Mahalanobis distance
        d = np.sqrt(y.T @ np.linalg.inv(S) @ y)
        if d < self.outlier_threshold:
            # If not an outlier, update the state
            self.x = self.x + K @ y
            self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        else:
            print("Outlier detected:", z)


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


def mahalanobis_distance(y, S):
    return np.sqrt(y.T @ np.linalg.inv(S) @ y)


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
    kalman = Custom_EKF(dt=delta_t)
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

        # Kalman filter prediction
        kalman.predict()

        # Kalman filter prediction
        kalman.predict()

        # Calculate the innovation (measurement residual)
        predicted_state = kalman.x
        innovation = np.hstack((tvec.reshape(-1), np.zeros(3))) - predicted_state
        S = (
            kalman.jacobian_H(predicted_state)
            @ kalman.P
            @ kalman.jacobian_H(predicted_state).T
            + kalman.R
        )
        mahalanobis_dist = mahalanobis_distance(innovation, S)

        if mahalanobis_dist < outlier_threshold:
            print("Not an outlier")
            kalman.update(np.hstack((tvec.reshape(-1), np.zeros(3))))
            predicted_state = kalman.x

            old_gray_frame = gray_frame.copy()
            old_points = (
                good_new_points.reshape(-1, 1, 2)
                if good_new_points.size > 0
                else old_points
            )

            imgpts, _ = cv2.projectPoints(
                axis, rvec, predicted_state[4:6].reshape(-1, 1), K, dist_coeffs
            )

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
