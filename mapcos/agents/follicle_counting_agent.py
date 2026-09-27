"""
Follicle Counting Agent
--------------------------
Classical CV, no trained model. Detects follicles via Hough Circle Transform.
Outputs follicle count + (x, y, r) locations, which the Grounding Agent will
turn into a binary mask to compare against the CNN's Grad-CAM region.
"""

import cv2
import numpy as np

from mapcos.config import HOUGH_PARAMS, PCOM_THRESHOLD


class FollicleCountingAgent:
    def __init__(self, dp=None, min_dist=None, param1=None, param2=None,
                 min_radius=None, max_radius=None, pcom_threshold: int = PCOM_THRESHOLD):
        p = HOUGH_PARAMS
        self.dp = dp or p["dp"]
        self.min_dist = min_dist or p["min_dist"]
        self.param1 = param1 or p["param1"]
        self.param2 = param2 or p["param2"]
        self.min_radius = min_radius or p["min_radius"]
        self.max_radius = max_radius or p["max_radius"]
        self.pcom_threshold = pcom_threshold  # Rotterdam PCOM: >=20 follicles per ovary

    def run(self, image_path: str) -> dict:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.equalizeHist(gray)  # boosts contrast on noisy ultrasound frames

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=self.dp, minDist=self.min_dist,
            param1=self.param1, param2=self.param2,
            minRadius=self.min_radius, maxRadius=self.max_radius
        )

        locations = []
        annotated = img_bgr.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for (x, y, r) in circles:
                locations.append({"x": int(x), "y": int(y), "r": int(r)})
                cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
                cv2.circle(annotated, (x, y), 2, (0, 0, 255), 3)

        count = len(locations)
        return {
            "agent": "follicle_counting",
            "follicle_count": count,
            "locations": locations,
            "image_shape": img_bgr.shape[:2],
            "meets_pcom_threshold": count >= self.pcom_threshold,
            "annotated_image": annotated,
        }
