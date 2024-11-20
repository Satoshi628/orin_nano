import numpy as np

# -1: Unknown, 0: front, 1: right-front, 2: right-side, 3: right-back, 4: back, 5: left-back, 6: left-side, 7: left-front
NORMAL_VECTORS = [[0., 0., -1.],
                  [-1, 0., -1],
                  [-1., 0., 0.],
                  [-1, 0., 1],
                  [0., 0., 1.],
                  [1, 0., 1],
                  [1, 0., 0.],
                  [1, 0., -1],
                  [1., 0., 0.],]
NORMAL_VECTORS = np.array(NORMAL_VECTORS)
NORMAL_VECTORS = NORMAL_VECTORS/np.linalg.norm(NORMAL_VECTORS, axis=-1, keepdims=True)
NORMAL_VECTORS = NORMAL_VECTORS.tolist()

HEADDIRECTION_NAME = ["Front",
                      "R-Front",
                      "R-Side",
                      "R-Back",
                      "Back",
                      "L-Back",
                      "L-Side",
                      "L-Front",
                      "Unknown"]


NON_DATA = 0

MOVENET_LINES = [
    ('nose', 'r_sho'), ('nose', 'l_sho'),
    ('r_sho', 'r_elb'), ('r_elb', 'r_wri'),
    ('l_sho', 'l_elb'), ('l_elb', 'l_wri'),
    ('l_hip', 'l_sho'), ('r_hip', 'r_sho'),
    ('l_hip', 'r_hip'),
    ('r_hip', 'r_knee'), ('r_knee', 'r_ank'),
    ('l_hip', 'l_knee'), ('l_knee', 'l_ank'),
    ('nose', 'r_eye'), ('nose', 'l_eye'),
    ('r_eye', 'r_ear'), ('l_eye', 'l_ear')
] 

KEYPOINTS_IDX = {'nose': 0,
                          'l_eye': 1,
                          'r_eye': 2,
                          'l_ear': 3,
                          'r_ear': 4,
                          'l_sho': 5,
                          'r_sho': 6,
                          'l_elb': 7,
                          'r_elb': 8,
                          'l_wri': 9,
                          'r_wri': 10,
                          'l_hip': 11,
                          'r_hip': 12,
                          'l_knee': 13,
                          'r_knee': 14,
                          'r_ank': 15,
                          'l_ank': 16,
                          }


UNKNOWN = 0
Standing = 1
Sitting = 2
Decubitus = 3
STATUS_NAME2IDX = {"UNKNOWN": UNKNOWN, "Standing": Standing, "Sitting": Sitting, "Decubitus": Decubitus}
STATUS_IDX2NAME = ["UNKNOWN", "Standing", "Sitting", "Decubitus"]

SITUATION_COLOR = {
    "Standing": (0,0,255), #red
    "Sitting": (255,0,0), #blue
    "Decubitus": (0,255,0), #green
    "UNKNOWN": (255,255,255)
}
