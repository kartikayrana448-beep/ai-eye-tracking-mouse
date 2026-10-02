import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/face_landmarker.task"

SCREEN_W, SCREEN_H = pyautogui.size()

CALIBRATION_POINTS = [
    (0.10, 0.10),
    (0.50, 0.10),
    (0.90, 0.10),

    (0.10, 0.50),
    (0.50, 0.50),
    (0.90, 0.50),

    (0.10, 0.90),
    (0.50, 0.90),
    (0.90, 0.90),
]

# =========================================================
# SCROLL SETTINGS
# =========================================================

SCROLL_ZONE = 0.10
SCROLL_ACTIVATION_TIME = 0.5
SCROLL_REPEAT = 0.25

scroll_direction = 0
scroll_start = None
last_scroll_time = 0

# =========================================================
# EYE LANDMARKS
# =========================================================

LEFT_EYE = {
    "outer": 33,
    "inner": 133,
    "top": 159,
    "bottom": 145
}

RIGHT_EYE = {
    "outer": 362,
    "inner": 263,
    "top": 386,
    "bottom": 374
}

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_BLINK = {
    "left": 33,
    "right": 133,
    "top": 159,
    "bottom": 145
}

RIGHT_BLINK = {
    "left": 362,
    "right": 263,
    "top": 386,
    "bottom": 374
}

# =========================================================
# FACE LANDMARKS
# =========================================================

FACE_LEFT = 234
FACE_RIGHT = 454
FACE_TOP = 10
FACE_BOTTOM = 152
FACE_NOSE = 1

# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    landmarker.close()
    exit()

# =========================================================
# BASIC FUNCTIONS
# =========================================================

def point(landmarks, index):
    return np.array([
        landmarks[index].x,
        landmarks[index].y
    ], dtype=np.float64)


def iris_center(landmarks, indices):
    points = np.array([
        [
            landmarks[i].x,
            landmarks[i].y
        ]
        for i in indices
    ])

    return np.mean(points, axis=0)


# =========================================================
# EYE GAZE
# =========================================================

def get_eye_features(landmarks, eye, iris_indices):

    outer = point(
        landmarks,
        eye["outer"]
    )

    inner = point(
        landmarks,
        eye["inner"]
    )

    top = point(
        landmarks,
        eye["top"]
    )

    bottom = point(
        landmarks,
        eye["bottom"]
    )

    iris = iris_center(
        landmarks,
        iris_indices
    )

    horizontal_axis = inner - outer
    vertical_axis = bottom - top

    width_sq = np.dot(
        horizontal_axis,
        horizontal_axis
    )

    height_sq = np.dot(
        vertical_axis,
        vertical_axis
    )

    if width_sq < 1e-8:
        return None

    if height_sq < 1e-8:
        return None

    horizontal = np.dot(
        iris - outer,
        horizontal_axis
    ) / width_sq

    vertical = np.dot(
        iris - top,
        vertical_axis
    ) / height_sq

    return np.array([
        horizontal,
        vertical
    ])


# =========================================================
# HEAD FEATURES
# =========================================================

def get_head_features(landmarks):

    left = point(
        landmarks,
        FACE_LEFT
    )

    right = point(
        landmarks,
        FACE_RIGHT
    )

    top = point(
        landmarks,
        FACE_TOP
    )

    bottom = point(
        landmarks,
        FACE_BOTTOM
    )

    nose = point(
        landmarks,
        FACE_NOSE
    )

    face_width = np.linalg.norm(
        right - left
    )

    face_height = np.linalg.norm(
        bottom - top
    )

    if face_width < 1e-6:
        return None

    if face_height < 1e-6:
        return None

    head_x = (
        nose[0]
        - (left[0] + right[0]) / 2
    ) / face_width

    head_y = (
        nose[1]
        - (top[1] + bottom[1]) / 2
    ) / face_height

    face_size = face_width

    return np.array([
        head_x,
        head_y,
        face_size
    ])


# =========================================================
# COMPLETE GAZE FEATURES
# =========================================================

def get_features(landmarks):

    left_eye = get_eye_features(
        landmarks,
        LEFT_EYE,
        LEFT_IRIS
    )

    right_eye = get_eye_features(
        landmarks,
        RIGHT_EYE,
        RIGHT_IRIS
    )

    head = get_head_features(
        landmarks
    )

    if left_eye is None:
        return None

    if right_eye is None:
        return None

    if head is None:
        return None

    lx, ly = left_eye
    rx, ry = right_eye

    hx, hy, face_size = head

    return np.array([
        1.0,

        lx,
        ly,

        rx,
        ry,

        (lx + rx) / 2,
        (ly + ry) / 2,

        hx,
        hy,

        face_size,

        lx * lx,
        ly * ly,

        rx * rx,
        ry * ry,

        hx * hx,
        hy * hy,

        lx * rx,
        ly * ry,

        hx * lx,
        hx * rx,

        hy * ly,
        hy * ry
    ])


# =========================================================
# BLINK DETECTION
# =========================================================

def eye_open_ratio(landmarks, eye):

    left = point(
        landmarks,
        eye["left"]
    )

    right = point(
        landmarks,
        eye["right"]
    )

    top = point(
        landmarks,
        eye["top"]
    )

    bottom = point(
        landmarks,
        eye["bottom"]
    )

    width = np.linalg.norm(
        right - left
    )

    height = np.linalg.norm(
        bottom - top
    )

    if width < 1e-8:
        return 0.0

    return height / width


def get_blink_ratio(landmarks):

    left_ratio = eye_open_ratio(
        landmarks,
        LEFT_BLINK
    )

    right_ratio = eye_open_ratio(
        landmarks,
        RIGHT_BLINK
    )

    return (
        left_ratio + right_ratio
    ) / 2.0


# =========================================================
# STABLE CALIBRATION AVERAGE
# =========================================================

def stable_average(samples):

    samples = np.array(samples)

    if len(samples) < 10:
        return None

    median = np.median(
        samples,
        axis=0
    )

    distances = np.linalg.norm(
        samples - median,
        axis=1
    )

    threshold = np.percentile(
        distances,
        70
    )

    stable = samples[
        distances <= threshold
    ]

    if len(stable) < 5:
        return None

    return np.mean(
        stable,
        axis=0
    )


# =========================================================
# CALIBRATION
# =========================================================

print()
print("================================")
print("        EYE CALIBRATION")
print("================================")
print()
print("Look directly at each RED dot.")
print("Keep your head relatively still.")
print()

calibration_data = []

for number, (target_x, target_y) in enumerate(
    CALIBRATION_POINTS,
    start=1
):

    samples = []

    cv2.namedWindow(
        "Calibration",
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        "Calibration",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    start = time.time()

    while time.time() - start < 3.0:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            time.time() * 1000
        )

        result = landmarker.detect_for_video(
            image,
            timestamp
        )

        screen = np.zeros(
            (
                SCREEN_H,
                SCREEN_W,
                3
            ),
            dtype=np.uint8
        )

        px = int(
            target_x * SCREEN_W
        )

        py = int(
            target_y * SCREEN_H
        )

        cv2.circle(
            screen,
            (px, py),
            28,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            screen,
            (px, py),
            8,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            screen,
            f"Look at the dot   {number}/9",
            (50, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Calibration",
            screen
        )

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            features = get_features(
                landmarks
            )

            if features is not None:

                if time.time() - start > 0.7:

                    samples.append(
                        features
                    )

        key = cv2.waitKey(1)

        if key == ord("q"):

            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            exit()

    average = stable_average(
        samples
    )

    if average is not None:

        calibration_data.append(
            (
                average,
                np.array([
                    target_x,
                    target_y
                ])
            )
        )

        print(
            f"Point {number}: "
            f"Samples={len(samples)}"
        )

    else:

        print(
            f"Point {number}: "
            "UNSTABLE DATA"
        )


cv2.destroyWindow(
    "Calibration"
)

# =========================================================
# TRAIN MODEL
# =========================================================

if len(calibration_data) < 7:

    print()
    print("ERROR: Not enough calibration data.")
    print()

    cap.release()
    landmarker.close()
    exit()


X = np.array([
    item[0]
    for item in calibration_data
])

Y = np.array([
    item[1]
    for item in calibration_data
])

lambda_reg = 0.001

I = np.eye(
    X.shape[1]
)

I[0, 0] = 0

coefficients = np.linalg.solve(
    X.T @ X + lambda_reg * I,
    X.T @ Y
)

print()
print("================================")
print("      CALIBRATION COMPLETE")
print("================================")
print()

time.sleep(1)

# =========================================================
# REAL-TIME MODE
# =========================================================

print("================================")
print("       REAL-TIME MODE")
print("================================")
print()

print("LOOK           = MOVE")
print("BLINK          = LEFT CLICK")
print("DOUBLE BLINK   = RIGHT CLICK")
print("LOOK TOP       = SCROLL UP")
print("LOOK BOTTOM    = SCROLL DOWN")
print()
print("Real-time eye control is ACTIVE.")
print("Press CTRL+C in this terminal to stop.")
print()

# =========================================================
# GAZE STATE
# =========================================================

smooth_x = SCREEN_W / 2
smooth_y = SCREEN_H / 2

alpha = 0.12

last_features = None

# =========================================================
# BLINK STATE
# =========================================================

BLINK_THRESHOLD = 0.18

MIN_CLOSED_TIME = 0.08
MAX_CLOSED_TIME = 0.80

blink_start = None
eyes_closed = False

last_click_time = 0
CLICK_COOLDOWN = 0.8

# =========================================================
# DOUBLE BLINK
# =========================================================

DOUBLE_BLINK_WINDOW = 0.65

first_blink_time = None
waiting_for_second_blink = False

# =========================================================
# REAL-TIME LOOP
# =========================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            time.time() * 1000
        )

        result = landmarker.detect_for_video(
            image,
            timestamp
        )

        now = time.time()

        if not result.face_landmarks:
            continue

        landmarks = result.face_landmarks[0]

        # =================================================
        # GAZE
        # =================================================

        features = get_features(
            landmarks
        )

        if features is not None:

            if last_features is None:

                last_features = features

            else:

                last_features = (
                    0.25 * features
                    + 0.75 * last_features
                )

            prediction = (
                last_features @ coefficients
            )

            predicted_x = np.clip(
                prediction[0],
                0.0,
                1.0
            )

            predicted_y = np.clip(
                prediction[1],
                0.0,
                1.0
            )

            target_x = (
                predicted_x * SCREEN_W
            )

            target_y = (
                predicted_y * SCREEN_H
            )

            smooth_x = (
                alpha * target_x
                + (1 - alpha) * smooth_x
            )

            smooth_y = (
                alpha * target_y
                + (1 - alpha) * smooth_y
            )

            # =================================================
            # REAL CURSOR
            # =================================================

            pyautogui.moveTo(
                int(smooth_x),
                int(smooth_y),
                duration=0
            )

            # =================================================
            # SCROLL
            # =================================================

            normalized_y = (
                smooth_y / SCREEN_H
            )

            direction = 0

            if normalized_y < SCROLL_ZONE:

                direction = 1

            elif normalized_y > (
                1 - SCROLL_ZONE
            ):

                direction = -1

            if direction == 0:

                scroll_direction = 0
                scroll_start = None

            else:

                if direction != scroll_direction:

                    scroll_direction = direction
                    scroll_start = now

                elif (
                    scroll_start is not None
                    and now - scroll_start
                    >= SCROLL_ACTIVATION_TIME
                    and now - last_scroll_time
                    >= SCROLL_REPEAT
                    and not eyes_closed
                ):

                    pyautogui.scroll(
                        direction
                    )

                    last_scroll_time = now

        # =================================================
        # BLINK
        # =================================================

        blink_ratio = get_blink_ratio(
            landmarks
        )

        if blink_ratio < BLINK_THRESHOLD:

            if blink_start is None:

                blink_start = now

            closed_time = (
                now - blink_start
            )

            if (
                not eyes_closed
                and closed_time >= MIN_CLOSED_TIME
                and closed_time <= MAX_CLOSED_TIME
            ):

                if (
                    now - last_click_time
                    >= CLICK_COOLDOWN
                ):

                    # First blink
                    if not waiting_for_second_blink:

                        first_blink_time = now
                        waiting_for_second_blink = True

                        print("FIRST BLINK")

                    # Second blink
                    elif (
                        now - first_blink_time
                        <= DOUBLE_BLINK_WINDOW
                    ):

                        pyautogui.rightClick()

                        last_click_time = now

                        print(
                            "DOUBLE BLINK -> RIGHT CLICK"
                        )

                        waiting_for_second_blink = False
                        first_blink_time = None

                eyes_closed = True

        else:

            blink_start = None
            eyes_closed = False

        # =================================================
        # SINGLE BLINK TIMEOUT
        # =================================================

        if waiting_for_second_blink:

            if (
                now - first_blink_time
                > DOUBLE_BLINK_WINDOW
            ):

                if (
                    now - last_click_time
                    >= CLICK_COOLDOWN
                ):

                    pyautogui.click()

                    last_click_time = now

                    print(
                        "BLINK -> LEFT CLICK"
                    )

                waiting_for_second_blink = False
                first_blink_time = None

except KeyboardInterrupt:

    print()
    print("Stopping real-time eye control...")

finally:

    cap.release()
    landmarker.close()

    print("Program ended.")