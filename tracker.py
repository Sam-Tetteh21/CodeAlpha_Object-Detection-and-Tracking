"""
tracker.py
----------
A lightweight multi-object tracker using the "centroid tracking"
technique - a simple, well-known approach (similar in spirit to SORT,
which also tracks objects frame-to-frame using position).

Concepts explained:

- A "centroid" is just the center point of a bounding box:
  centroid_x = x + width/2, centroid_y = y + height/2

- The core idea: on every new frame, we have a fresh list of detected
  boxes with NO identity information - YOLO doesn't know "this is the
  same car as last frame". So we match each new detection to the
  closest EXISTING tracked object (by centroid distance). If a new
  detection is close enough to a known object's last position, we
  assume it's the same object and keep its ID. If it's far from
  everything we know about, it's a brand-new object and gets a new ID.

- Objects that vanish (no matching detection) for too many consecutive
  frames are "deregistered" (removed), since they've likely left the
  frame or were a one-off false detection.

This is simpler than SORT/DeepSORT (which use Kalman filters and, for
DeepSORT, an appearance-matching neural network) but demonstrates the
same core tracking concept: giving each detected object a persistent
ID across frames instead of treating every frame as brand new.
"""

from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 75):
        self.next_object_id = 0
        self.objects = OrderedDict()       # object_id -> centroid (x, y)
        self.disappeared = OrderedDict()   # object_id -> num consecutive frames missing

        # How many frames an object can go "missing" before we forget it
        self.max_disappeared = max_disappeared
        # How far (in pixels) a detection can be from a known object
        # and still be considered "the same object"
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, boxes):
        """
        boxes: list of (x, y, w, h) tuples for this frame's detections.
        Returns a dict {object_id: (x, y, w, h)} - each detection box
        paired with its persistent tracking ID.
        """
        if len(boxes) == 0:
            # No detections this frame - mark all existing objects as
            # having disappeared for one more frame.
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        input_centroids = np.zeros((len(boxes), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(boxes):
            input_centroids[i] = (int(x + w / 2), int(y + h / 2))

        if len(self.objects) == 0:
            # No existing tracked objects - register every detection as new.
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Distance between every existing object and every new detection
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Match closest pairs first: sort rows by their smallest distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue  # too far apart - not the same object

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            # Existing objects that had no matching detection this frame
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Detections that didn't match any existing object = new objects
            for col in unused_cols:
                self.register(input_centroids[col])

        # Build the id -> box mapping for this frame to return
        result = {}
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        for object_id, centroid in zip(object_ids, object_centroids):
            # Find which input box this centroid came from, to keep w/h
            distances = [np.linalg.norm(centroid - ic) for ic in input_centroids]
            closest_idx = int(np.argmin(distances))
            if distances[closest_idx] <= self.max_distance or len(self.objects) <= len(boxes):
                result[object_id] = boxes[closest_idx]
        return result
