"""
Updated SmartFlow Traffic Script 
functions -> Detection / Tracking / Congestion Metrics

Right now, it:
 I.     ingests NDOT video footage
 II. can optionally warp the view to be shown top-down using homography
 III. 	loads a YOLO detection model to detect vehicles in each frame
 IV.	utilizes a centroid tracker to assign unique IDs and maintain continuity across frames for detected objects
 V. 	counts vehicles crossing a horizontal line that can be moved around (bidirectional)
 VI. 	estimates speed
 VII. 	computes congestion metrics (queue length, occupancy, flow, PTP, density)
 VIII. 	displays the results/metrics
 VIIII.	outputs the results to the ESP32 for further processing in JSON format

important functions:
  - CentroidTracker: a centroid-based tracker
  - mouse_callback: allows us to drag the boundary line around
  - main: loops detection -> tracking -> metric computation -> display -> output

  # added an homography transformation in the code, which converts our angled camera view into a top down view using the power of math and pixels
  # currently commented out all of it because im not sure what it will do or if it makes our metrics more accurate
  # (it is supposed to)
  
  # functions with or without homography (H)
  
  # w/o H: Everything runs on the original frame 
  # polygon ROIs (if exists) must be defined in original coordinates 
  # density/PTP still work
  # speed/queue in meters require a scale for the original frame
  
  # w/ H: Inference runs on the warped frame
  # overlays show side-by-side 
  # ROIs should be defined in warped coordinates 
  # scale applies to the warped plane
"""

#includes
import os
import sys
import argparse
import glob
import time
import json
import math
import signal
import requests
import serial
import csv
import cv2
import numpy as np

from ultralytics import YOLO

# definitions that include access to the NDOT REST API for traffic footage
# defines bounding box that geographically defines a rough Las Vegas region
MANIFEST_PATH = "ndot_manifest.json"
NDOT_CAM_URL = "https://www.nvroads.com/api/v2/get/cameras"
DEFAULT_BBOX = "36.00,36.35,-115.35,-115.00"  

THRESHOLD_DENSITY = 0.10
PTP_BUFFER = 0.02

# helper for exiting
def graceful_exit(sig, frame):
    print("\n[SmartFlow] Exiting...")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)

# helpers
# helper that parses through the bounding box "bbox"
def parse_bbox(s):
    lat_min, lat_max, lon_min, lon_max = map(float, s.split(","))
    return lat_min, lat_max, lon_min, lon_max

def in_bbox(lat, lon, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

# helper that loads a previously saved ndot path if it exists
def load_manifest(path=MANIFEST_PATH):
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

# helper that will save the ndot info if specified
def save_manifest(manifest, path=MANIFEST_PATH):
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

# helper that returns the signed value to know which side of the line that an object crossed        
def signed_side(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


# helper that uses our developer API key to access NDOT cameras
def fetch_ndot_cameras(api_key):
    if requests is None:
        raise RuntimeError("requests not available.")
    params = {"key": api_key, "format": "json"}
    r = requests.get(NDOT_CAM_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# helper that uses bbox or cam_id to access a camera feed
def choose_ndot_camera(cams, bbox, force_id=None):
    for c in cams:
        # if an id is specified in args, honor it (bbox input is ignored)
        if force_id is not None and str(c.get("Id")) != str(force_id):
            continue

        lat, lon = c.get("Latitude"), c.get("Longitude")

        # else apply the bbox and access the first cam we can get inside of it
        if force_id is None:
            if lat is None or lon is None or not in_bbox(lat, lon, bbox):
                continue

        # when accessing is a success, return these attributes
        for v in c.get("Views", []):
            url = v.get("VideoUrl", "") or ""
            if v.get("Status") == "Enabled" and url.endswith(".m3u8"):
                return {
                    "id": str(c.get("Id")),
                    "name": c.get("Roadway") or c.get("Name") or f"Camera {c.get('Id')}",
                    "lat": float(lat) if lat is not None else 0.0,
                    "lon": float(lon) if lon is not None else 0.0,
                    "url": url
                }
    return None
    
    #helper to detect if a centroid is in ANY ROI
def centroid_in_any_roi(cx, cy, roi_masks):
    for m in roi_masks.values():
        H, W = m.shape
        if 0 <= cy < H and 0 <= cx < W and m[cy, cx]:
            return True
    return False
    
# helper to open capture source
def open_capture_for_source(args, resize_dims):
    img_ext = {'.jpg', '.jpeg', '.png', '.bmp'} # create list of img formats
    vid_ext = {'.avi', '.mp4', '.mov', '.mkv', '.wmv'} # create list of video formats

    source_arg = args.source.lower()
    if os.path.isdir(source_arg):
        return 'image_folder', [os.path.join(source_arg, f) for f in os.listdir(source_arg)
                                if os.path.splitext(f)[1].lower() in img_ext], None, None
    elif os.path.isfile(source_arg):
        _, ext = os.path.splitext(source_arg)
        if ext.lower() in img_ext:
            return 'image_single', [source_arg], None, None
        elif ext.lower() in vid_ext:
            cap = cv2.VideoCapture(source_arg, cv2.CAP_ANY)
            return 'video', None, cap, None
        else:
            raise ValueError("Unsupported file type")
    elif source_arg.startswith('usb'):
        usb_idx = int(source_arg[3:])
        cap = cv2.VideoCapture(usb_idx, cv2.CAP_ANY)
        return 'usb', None, cap, None
    elif source_arg == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        size = resize_dims if resize_dims else (1280, 720)
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": size}))
        cap.start()
        return 'picamera', None, cap, None
    # added new ndot source
    elif source_arg == 'ndot':
        api_key = args.ndot_key or os.getenv("NDOT_API_KEY")
        if not api_key:
            raise ValueError("provide --ndot_key")
        bbox = parse_bbox(args.bbox)
        cams = fetch_ndot_cameras(api_key)
        chosen = choose_ndot_camera(cams, bbox, args.cam_id)
        if not chosen:
            raise RuntimeError("No camera found in bbox or cam_id not available.")
        print(f"[NDOT] Using camera {chosen['id']} {chosen['name']} @ ({chosen['lat']:.4f},{chosen['lon']:.4f})")
        cap = cv2.VideoCapture(chosen['url'], cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError("FFmpeg not avail.")
        return 'ndot', None, cap, chosen
    else:
        raise ValueError("Invalid source")

# CentroidTracker class
# maintains a mapping of unique object IDs to centroid positions
# if an obj disappears for > max_disappeared frames, it is deregistered
# reference: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
class CentroidTracker:
    def __init__(self, max_disappeared=90): #define number of max_disappeared frames and initialize constructor - changed to 10 from 50
        self.next_object_id = 0		#integer that is assigned to newest tracked object
        self.objects = {}         # object_id == (cx, cy), or a mapping of the most recent centroid to (cx, cy)
        self.disappeared = {}     # mapping of objects ID to number of consecutive missed frames.
        self.speed = {}  # new b/c it is added for storing per-object speeds
        self.max_disappeared = max_disappeared # when obj exceeds max_disappeared, we drop the object.

    def register(self, centroid): # define an object called register (in the context of python, not to be confused with objects we detect in YOLO), this object creates a new tracked object
        self.objects[self.next_object_id] = centroid # store the centroid in self.objects with a unique ID along with it
        self.disappeared[self.next_object_id] = 0 # set its disappeared counter to 0
        self.next_object_id += 1 # increment next_object_id for the next new obj

    def deregister(self, object_id): # define an object called deregister, which will remove an object when its not seen for too long
        del self.objects[object_id] # delete the obj ID in self.objects
        del self.disappeared[object_id] # delete the obj ID in self.disappeared
        if object_id in self.speed:
            del self.speed[object_id]

    def update(self, input_centroids): # define an object called update, which takes newly detected centroids for the current frame and returns the updated mapping object_id -> centroid

        if len(input_centroids) == 0: # check if there is no detection of an object in this frame
            for obj_id in list(self.disappeared.keys()): # for every tracked object
                self.disappeared[obj_id] += 1 # start incrementing all tracked objects disappeared counter
                if self.disappeared[obj_id] > self.max_disappeared: # deregister objects -> disappeared > 10
                    self.deregister(obj_id)
            return self.objects # return the current objects mapping with no new assignments

        # if nothing is being tracked yet, register every detection as a new object
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else: # o/w prepare lists of current tracked IDs and centroids for matching (object_ids[i] == object_centroids[i])
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # this lines computes an m x n distance matrix 'D', where m == number of tracked obj, n == number of new detections
	    # purpose - process objects starting from those w/ the smallest nearest neighbor distance to reduce chances of mismatches
            D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids, axis=2) # maps old centroids into (m,1,2) and input_centroids to (n,2) so broadcasting gives an (m,n,2) difference
	    # this gives us (m,n) Euclidean distances
            rows = D.min(axis=1).argsort() # min(axis=1) gives min distance (closest detection) for each tracked object. argsort() sorts object rows by their nearest neighbor distances
            cols = D.argmin(axis=1)[rows] #argmin returns column index (detection) of the closest detection for each obj, indexing it w/ [rows] reorders those chosen indices in the order of rows.


	    # this part iterates the row/col pairs and greedily assign them (smallest neighbor distance first)
            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols): # for every pair
                if row in used_rows or col in used_cols: # continue or skip if the obj row or column was already assigned
                    # do this so that each obj/detection participates in at most one match
                    continue
                obj_id = object_ids[row] # update the tracked objs centroid to the matched detection
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0 # reset counter
                used_rows.add(row) # mark the row used
                used_cols.add(col) # mark the col used
                # defined as a greedy matching method according to the internet but simple to implement for our case

            # any detection (col) that was not matched to an existing obj becomes a new obj, so we register it
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

            # any tracked object (row) that was not matched to a detection is 'missing' in this frame
            unused_rows = set(range(len(object_ids))) - used_rows
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1 # increment its disappeared counter
                if self.disappeared[obj_id] > self.max_disappeared: # ready to dereg it if desired
                    self.deregister(obj_id)

        return self.objects # return the mapping of object_id -> (cx, cy)

# homography calibration ui for if we pass --calibrate in as 
# click 4 road-plane points on the view
# computes H mapping to a rect w/ size dst_size
class HomographyCalibrator:
    def __init__(self, window_name="Homography Calib",
                 max_size="1600x900",
                 force_scale=None,
                 resize_dims=None):
        self.window_name = window_name
        self.points = []   # saved in ORIGINAL frame coords (after any resize)
        self.dst_size = (640, 480)
        self.finished = False
        self.force_scale = force_scale
        self.resize_dims = resize_dims

        # parse max display size
        try:
            w, h = map(int, max_size.lower().split('x'))
            self.max_w, self.max_h = w, h
        except Exception:
            self.max_w, self.max_h = 1600, 900

        self._disp_scale = 1.0
        self._last_disp_shape = None

    def _compute_scale(self, frame_w, frame_h):
        if self.force_scale is not None:
            return float(self.force_scale)
        sw = self.max_w / frame_w
        sh = self.max_h / frame_h
        return min(sw, sh, 3.0)  # cap so we don't over-blow

    def _to_disp(self, frame):
        h, w = frame.shape[:2]
        if self._last_disp_shape is None:
            self._disp_scale = self._compute_scale(w, h)
        if abs(self._disp_scale - 1.0) < 1e-3:
            return frame
        disp = cv2.resize(frame, (int(w*self._disp_scale), int(h*self._disp_scale)), interpolation=cv2.INTER_LINEAR)
        self._last_disp_shape = disp.shape
        return disp

    def _disp_to_orig(self, x, y):
        # map our click from display space back to original frame
        s = max(self._disp_scale, 1e-6)
        return int(round(x / s)), int(round(y / s))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            ox, oy = self._disp_to_orig(x, y)
            self.points.append((ox, oy))
            print(f"[Calib] Added point {len(self.points)} (orig): {ox}, {oy}")

    def run(self, cap, cam_key, manifest):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)   # resizable
        cv2.moveWindow(self.window_name, 60, 40)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        print("[Calib] Click 4 road-plane points. Keys: [r]eset  [s]ave  [q]uit")

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
                
            if self.resize_dims is not None:
                w_des, h_des = self.resize_dims
                h_f, w_f = frame.shape[:2]
                if (w_f, h_f) != (w_des, h_des):
                    frame = cv2.resize(frame, (w_des, h_des))

            disp = self._to_disp(frame).copy()

            # rraw already-added points (convert orig to disp for preview)
            for i, (ox, oy) in enumerate(self.points):
                dx, dy = int(round(ox*self._disp_scale)), int(round(oy*self._disp_scale))
                cv2.circle(disp, (dx, dy), 6, (0,255,255), -1)
                cv2.putText(disp, f"P{i+1}", (dx+8, dy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.putText(disp, "Calibrate homography: [r]=reset  [s]=save  [q]=quit",
                        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow(self.window_name, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                self.points = []
                print("[Calib] Reset points.")
            elif k == ord('s'):
                if len(self.points) == 4:
                    src_pts = np.float32(self.points)
                    w, h = self.dst_size
                    dst_pts = np.float32([[0, h], [w, h], [w, 0], [0, 0]])
                    H, _ = cv2.findHomography(src_pts, dst_pts)
                    if H is None:
                        print("[Calib] Homography failed. Try again.")
                        continue
                    manifest.setdefault(cam_key, {})
                    manifest[cam_key]["homography"] = {
                        "src": self.points,               # ORIGINAL coords
                        "dst": [[0, h], [w, h], [w, 0], [0, 0]],
                        "warp_size": [w, h]
                    }
                    save_manifest(manifest)
                    print(f"[Calib] Saved homography for '{cam_key}' to {MANIFEST_PATH}")
                    self.finished = True
                    break
                else:
                    print("[Calib] Need exactly 4 points.")

        cv2.destroyWindow(self.window_name)
        return self.finished

class ROICalibrator:

      # n  finish current ROI and start a new one
      # u  undo last point in current ROI
      # c  clear all ROIs
      # s  save all ROIs to manifest and exit
      # q  quit w/o saving
      
    def __init__(self, window_name="ROI Calib", max_size="1600x900",
                use_homography=False, H=None, warp_size=None,
                force_scale=None, resize_dims=None):
        self.window_name = window_name
        self.points = []       # current polygon points
        self.force_scale = force_scale
        self.resize_dims = resize_dims
        self.rois = {}         # name mapped to list[(x,y)]
        self.next_roi_idx = 1

        self.use_homography = use_homography
        self.H = H
        self.warp_size = warp_size

        self.finished = False
        self.force_scale = force_scale

        # parse max display size
        try:
            w, h = map(int, max_size.lower().split('x'))
            self.max_w, self.max_h = w, h
        except Exception:
            self.max_w, self.max_h = 1600, 900

        self._disp_scale = 1.0
        self._last_disp_shape = None

    def _compute_scale(self, frame_w, frame_h):
        if self.force_scale is not None:
            return float(self.force_scale)
        sw = self.max_w / frame_w
        sh = self.max_h / frame_h
        return min(sw, sh, 3.0)

    def _to_disp(self, frame):
        h, w = frame.shape[:2]
        if self._last_disp_shape is None:
            self._disp_scale = self._compute_scale(w, h)
        if abs(self._disp_scale - 1.0) < 1e-3:
            return frame
        disp = cv2.resize(frame, (int(w * self._disp_scale), int(h * self._disp_scale)),
                          interpolation=cv2.INTER_LINEAR)
        self._last_disp_shape = disp.shape
        return disp

    def _disp_to_orig(self, x, y):
        s = max(self._disp_scale, 1e-6)
        return int(round(x / s)), int(round(y / s))

    def _prepare_frame(self, frame):
        # apply homography if ROIs should be in warped plane
        if self.use_homography and self.H is not None and self.warp_size is not None:
            warp_w, warp_h = self.warp_size
            frame = cv2.warpPerspective(frame, self.H, (warp_w, warp_h),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT)
        return frame

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = self._disp_to_orig(x, y)
            self.points.append((ox, oy))
            print(f"[ROI Calib] added point {len(self.points)} for current ROI: ({ox}, {oy})")

    def _finalize_current_roi(self):
        if len(self.points) >= 3:
            name = f"ROI{self.next_roi_idx}"
            self.rois[name] = list(self.points)
            print(f"[ROI Calib] finalized {name} with {len(self.points)} points.")
            self.next_roi_idx += 1
            self.points = []
        else:
            print("[ROI Calib] need at least 3 points to form an ROI.")

    def run(self, cap, cam_key, manifest):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, 80, 60)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        print("[ROI Calib] instructions:")
        print("  - click to add vertices for the current ROI.")
        print("  - [n] finish current ROI and start a new one.")
        print("  - [u] undo last point in current ROI.")
        print("  - [c] clear ALL ROIs.")
        print("  - [s] save ROIs to manifest and exit.")
        print("  - [q] quit WITHOUT saving.")

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
                
            if self.resize_dims is not None:
                w_des, h_des = self.resize_dims
                h_f, w_f = frame.shape[:2]
                if (w_f, h_f) != (w_des, h_des):
                    frame = cv2.resize(frame, (w_des, h_des))

            frame_proc = self._prepare_frame(frame)
            disp = self._to_disp(frame_proc).copy()

            # draw finished ROIs
            for name, pts in self.rois.items():
                if len(pts) >= 3:
                    pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    # project to display space
                    pts_disp = (pts_arr * self._disp_scale).astype(int)
                    cv2.polylines(disp, [pts_disp], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(disp, name,
                                tuple(pts_disp[0, 0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # draw current points
            if self.points:
                pts_arr = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)
                pts_disp = (pts_arr * self._disp_scale).astype(int)
                for p in pts_disp:
                    cv2.circle(disp, tuple(p[0]), 4, (0, 255, 255), -1)
                if len(pts_disp) > 1:
                    cv2.polylines(disp, [pts_disp], isClosed=False, color=(0, 255, 255), thickness=2)

            cv2.putText(disp, "ROI Calib: [n]=new ROI  [u]=undo  [c]=clear  [s]=save  [q]=quit",
                        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(self.window_name, disp)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                print("[ROI Calib] quit without saving.")
                break
            elif k == ord('n'):
                self._finalize_current_roi()
            elif k == ord('u'):
                if self.points:
                    removed = self.points.pop()
                    print(f"[ROI Calib] removed point {removed} from current ROI.")
            elif k == ord('c'):
                self.points = []
                self.rois = {}
                self.next_roi_idx = 1
                print("[ROI Calib] cleared all ROIs.")
            elif k == ord('s'):
                if self.points:
                    self._finalize_current_roi()
                if not self.rois:
                    print("[ROI Calib] no ROIs to save.")
                    continue
                manifest.setdefault(cam_key, {})
                manifest[cam_key]["rois"] = self.rois
                save_manifest(manifest)
                print(f"[ROI Calib] saved {len(self.rois)} ROIs for '{cam_key}' to {MANIFEST_PATH}")
                self.finished = True
                break

        cv2.destroyWindow(self.window_name)
        return self.finished
        
class LineCalibrator:
#line calibrator class, lets us create our own dynamic lines for crossing counts
    def __init__(self, window_name="Line Calib", max_size="1600x900",
                 use_homography=False, H=None, warp_size=None,
                 force_scale=None, resize_dims=None):
        self.window_name = window_name
        self.points = []    # current line
        self.force_scale = force_scale
        self.resize_dims = resize_dims
        self.lines = {}     # name mapped to [(x1,y1),(x2,y2)]
        self.next_idx = 1
        self.finished = False

        self.use_homography = use_homography
        self.H = H
        self.warp_size = warp_size
        self.force_scale = force_scale

        try:
            w, h = map(int, max_size.lower().split('x'))
            self.max_w, self.max_h = w, h
        except Exception:
            self.max_w, self.max_h = 1600, 900

        self._disp_scale = 1.0
        self._last_disp_shape = None

    def _compute_scale(self, frame_w, frame_h):
        if self.force_scale is not None:
            return float(self.force_scale)
        sw = self.max_w / frame_w
        sh = self.max_h / frame_h
        return min(sw, sh, 3.0)

    def _prepare_frame(self, frame):
        if self.use_homography and self.H is not None and self.warp_size is not None:
            warp_w, warp_h = self.warp_size
            frame = cv2.warpPerspective(frame, self.H, (warp_w, warp_h),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT)
        return frame

    def _to_disp(self, frame):
        h, w = frame.shape[:2]
        if self._last_disp_shape is None:
            self._disp_scale = self._compute_scale(w, h)
        if abs(self._disp_scale - 1.0) < 1e-3:
            return frame
        disp = cv2.resize(frame,
                          (int(w * self._disp_scale), int(h * self._disp_scale)),
                          interpolation=cv2.INTER_LINEAR)
        self._last_disp_shape = disp.shape
        return disp

    def _disp_to_orig(self, x, y):
        s = max(self._disp_scale, 1e-6)
        return int(round(x / s)), int(round(y / s))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = self._disp_to_orig(x, y)
            self.points.append((ox, oy))
            print(f"[Line Calib] point {len(self.points)}: ({ox},{oy})")

    def _finalize_line(self):
        if len(self.points) == 2:
            name = f"L{self.next_idx}"
            self.lines[name] = list(self.points)
            print(f"[Line Calib] finalized {name}: {self.points}")
            self.points = []
            self.next_idx += 1
        else:
            print("[Line Calib] need exactly 2 points for a line.")

    def run(self, cap, cam_key, manifest):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, 80, 40)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        print("[Line Calib] click 2 points per line.")
        print("  [n] new line  [u] undo  [c] clear  [s] save  [q] quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
                
            if self.resize_dims is not None:
                w_des, h_des = self.resize_dims
                h_f, w_f = frame.shape[:2]
                if (w_f, h_f) != (w_des, h_des):
                    frame = cv2.resize(frame, (w_des, h_des))

            frame_proc = self._prepare_frame(frame)
            disp = self._to_disp(frame_proc).copy()

            # draw finished lines
            for name, pts in self.lines.items():
                (x1, y1), (x2, y2) = pts
                p1 = (int(x1 * self._disp_scale), int(y1 * self._disp_scale))
                p2 = (int(x2 * self._disp_scale), int(y2 * self._disp_scale))
                cv2.line(disp, p1, p2, (0, 255, 255), 2)
                cv2.putText(disp, name, (p1[0] + 5, p1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # draw provisional line
            if self.points:
                pts = [(int(px * self._disp_scale), int(py * self._disp_scale))
                       for (px, py) in self.points]
                for p in pts:
                    cv2.circle(disp, p, 5, (0, 255, 0), -1)
                if len(pts) == 2:
                    cv2.line(disp, pts[0], pts[1], (0, 255, 0), 2)

            cv2.putText(disp,
                        "Line Calib: click-click=line  [n]=new  [u]=undo  [c]=clear  [s]=save  [q]=quit",
                        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(self.window_name, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                print("[Line Calib] quit without saving.")
                break
            elif k == ord('n'):
                self._finalize_line()
            elif k == ord('u'):
                if self.points:
                    self.points.pop()
            elif k == ord('c'):
                self.points = []
                self.lines = {}
                self.next_idx = 1
                print("[Line Calib] cleared all lines.")
            elif k == ord('s'):
                if self.points:
                    self._finalize_line()
                if not self.lines:
                    print("[Line Calib] no lines to save.")
                    continue
                manifest.setdefault(cam_key, {})
                manifest[cam_key]["lines"] = self.lines
                save_manifest(manifest)
                print(f"[Line Calib] saved {len(self.lines)} lines for '{cam_key}'")
                self.finished = True
                break

        cv2.destroyWindow(self.window_name)
        return self.finished


# helper that returns the list of line attributes from the manifest
def load_lines_for_cam(manifest, cam_key):
    out = []
    lines_dict = manifest.get(cam_key, {}).get("lines", {})
    for name, pts in lines_dict.items():
        if len(pts) != 2:
            continue
        (x1, y1), (x2, y2) = pts
        out.append({
            "name": name,
            "p1": (int(x1), int(y1)),
            "p2": (int(x2), int(y2)),
        })
    return out


# helper that rebuilds the homograpy entry and rebuilds H + (warp_w, warp_h)
def load_homography_for_cam(manifest, cam_key):
    entry = manifest.get(cam_key, {}).get("homography")
    if not entry:
        return None, None
    src_pts = np.float32(entry["src"])
    dst_pts = np.float32(entry["dst"])
    warp_w, warp_h = map(int, entry["warp_size"])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H, (warp_w, warp_h)

# a binary mask for polygon ROIs, one mask per ROI
def polygon_mask(shape_hw, polygon_pts):
    # return uint8 mask with polygon filled (1s inside). shape_hw = (H, W).
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(polygon_pts) >= 3:
        pts = np.array(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask
    
# build the masks
def build_roi_masks(shape_hw, roi_polygons):   
    masks = {}
    areas = {}
    for name, poly in roi_polygons.items():
        m = polygon_mask(shape_hw, poly)
        a = int(m.sum())
        if a == 0:
            # avoid zero-area masks
            continue
        masks[name] = m
        areas[name] = a
    return masks, areas

# produces a union mask from multiple detected bounding boxes
def rect_union_mask(shape_hw, rects):
    H, W = shape_hw
    m = np.zeros((H, W), dtype=np.uint8)
    if not rects:
        return m
    for (x1,y1,x2,y2) in rects:
        x1 = max(0, min(W-1, int(x1))); x2 = max(0, min(W, int(x2)))
        y1 = max(0, min(H-1, int(y1))); y2 = max(0, min(H, int(y2)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1
    return m
    
def which_roi_contains(cx, cy, roi_masks):
    for name, m in roi_masks.items():
        H, W = m.shape
        if 0 <= cy < H and 0 <= cx < W and m[cy, cx]:
            return name
    return None

# tries to find meters-per-pixel scale in this order below
def compute_scale_m_per_px(args, manifest, cam_key, frame_shape_hw):
    # --scale_m_per_px
    # --scale_from_points (x1,y1,x2,y2,meters)
    # manifest[cam_key]['scale_m_per_px']
    # none
    
    # explicit override
    if args.scale_m_per_px is not None:
        return float(args.scale_m_per_px)

    # from two points and known length
    if args.scale_from_points:
        try:
            x1,y1,x2,y2,m = map(float, args.scale_from_points.split(','))
            dx, dy = x2 - x1, y2 - y1
            px = max(1e-6, math.hypot(dx, dy))
            return float(m / px)
        except Exception:
            print("[Scale] Invalid --scale_from_points format. Expected 'x1,y1,x2,y2,meters'.")

    # from manifest if it exists
    sc = manifest.get(cam_key, {}).get("scale_m_per_px")
    if sc is not None:
        try:
            return float(sc)
        except Exception:
            pass

    # none
    return None

# the main loop
# reads next frame from input source
# runs YOLO model to detect vehicles
# computes centroid of each bounding box for tracking
# uses CentroidTracker to maintain consistent obj IDs
# detects line crossing to count cars moving up/down
# annotates the video feed w/ bounding boxes, FPS, and more
# reports the gathered data
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--thresh', default=0.5, type=float)
    ap.add_argument('--resolution', default=None)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--warp_size', default="1280x720")
                
    #small window issue
    ap.add_argument('--display_max', default="1600x900")
    ap.add_argument('--display_scale', default=None, type=float)

    # metric accuracy upgrades
    ap.add_argument('--roi_mode', choices=['manifest','halves'], default='manifest',
                    help="use ROI polygons from manifest if it exists, else fallback to halves")
    ap.add_argument('--ptp_min_sec', type=float, default=5.0,
                    help="min duration (s) above threshold to record a PTP")
    ap.add_argument('--scale_m_per_px', type=float, default=None,
                    help="meters-per-pixel override")
    ap.add_argument('--scale_from_points', default=None,
                    help="compute scale from two points and known length: 'x1,y1,x2,y2,meters'. Example: 100,500,900,500,90")
    ap.add_argument('--queue_rois', default=None,
                    help="comma-separated ROI names to use for queue/dwell; default: all ROIs")
                
    # NDOT stuff
    ap.add_argument('--ndot_key', default=os.getenv('NDOT_API_KEY'))
    ap.add_argument('--bbox', default=DEFAULT_BBOX)
    ap.add_argument('--cam_id', default=None)

    # UART and reporting
    ap.add_argument('--interval', default=120, type=int, help="Aggregation interval seconds (e.g., 120/180)")
    ap.add_argument('--uart', default='/dev/serial0')
    ap.add_argument('--baud', default=115200, type=int)
    
     # homography calibration
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--force_use_homography', action='store_true')

    # ROI calibration (draw N ROIs)
    ap.add_argument('--roi_calibrate', action='store_true')
    
    ap.add_argument('--line_calibrate', action='store_true')
    
    ap.add_argument('--night_mode', action='store_true')
    
    ap.add_argument('--ignore_rois', default=None)
    
    args = ap.parse_args()

    # load model
    model = YOLO(args.model, task='detect')
    labels = model.names
    min_thresh = args.thresh

    # set resolution
    resize = False
    resW = resH = None
    if args.resolution:
        resize = True
        resW, resH = map(int, args.resolution.split('x'))
        resize_dims = (resW, resH)
    else:
        resize_dims = None

    # open the source
    source_type, imgs_list, cap, chosen_cam = open_capture_for_source(args, resize_dims)

    # uart
    ser = None
    if serial is not None:
        try:
            ser = serial.Serial(args.uart, args.baud, timeout=1)
        except Exception as e:
            print(f"[UART] Could not open {args.uart}: {e}")

    # attempt to load a manifest
    manifest = load_manifest()
    cam_key = chosen_cam['id'] if chosen_cam else "generic"
    H = None
    warp_size = None

    # if we pass in calibrate, go here
    if args.calibrate and source_type in ('ndot', 'video', 'usb', 'picamera'):
        # cannot use picam for this
        if source_type == 'picamera':
            print("use ndot or video for calibration.")
            sys.exit(1)
            
        # parse desired warp canvas and pass it to the calibrator
        try:
            wW, wH = map(int, args.warp_size.lower().split('x'))
        except Exception:
            wW, wH = 1280, 720
        
        calib = HomographyCalibrator(
            window_name="Homography Calib",
            max_size=args.display_max,
            force_scale=float(args.display_scale) if args.display_scale else None,
            resize_dims=resize_dims if resize else None
        )
        calib.dst_size = (wW, wH)   # should be a bigger, cleaner top-down which should help what we are seeing
        
        ok = calib.run(cap, cam_key, manifest)
        # if we are in the middle of doing homography but realize we do not want to do it anymore
        if not ok:
            print("[Calib] Not saved. Exiting.")
            sys.exit(0)
        # reload homography
        manifest = load_manifest()
        H, warp_size = load_homography_for_cam(manifest, cam_key)

    # normal run, try to load existing homography
    if H is None:
        H, warp_size = load_homography_for_cam(manifest, cam_key)

    use_homography = (H is not None) # enable if available by default
    if use_homography:
        print(f"[Homography] enabled for '{cam_key}'. warp size={warp_size}")
    else:
        print("[Homography] not enabled, using original frames.")

    # ROI calibration mode, draw polygons on the same plane used for metrics
    if args.roi_calibrate and source_type in ('ndot', 'video', 'usb', 'picamera'):
        roi_calib = ROICalibrator(
            window_name="ROI Calib",
            max_size=args.display_max,
            use_homography=use_homography,
            H=H,
            warp_size=warp_size,
            force_scale=float(args.display_scale) if args.display_scale else None,
            resize_dims=resize_dims if resize else None
        )
        ok = roi_calib.run(cap, cam_key, manifest)
        if not ok:
            print("[ROI Calib] Not saved. Exiting.")
            sys.exit(0)
        # reload ROIs for the actual run
        manifest = load_manifest()
        
    # === Line calibration (per-lane / per-approach lines) ===
    if args.line_calibrate and source_type in ('ndot', 'video', 'usb', 'picamera'):
        line_calib = LineCalibrator(
            window_name="Line Calib",
            max_size=args.display_max,
            use_homography=use_homography,
            H=H,
            warp_size=warp_size,
            force_scale=float(args.display_scale) if args.display_scale else None,
            resize_dims=resize_dims if resize else None
        )
        ok = line_calib.run(cap, cam_key, manifest)
        if not ok:
            print("[Line Calib] Not saved. Exiting.")
            sys.exit(0)
        # reload manifest so main run sees the saved lines
        manifest = load_manifest()

    # trackers and counters
    tracker = CentroidTracker(max_disappeared=90)
    prev_positions = {}
    frame_rate_buffer, fps_avg_len = [], 30

    counted_ids_down, counted_ids_up = set(), set()
    car_counter_down = 0
    car_counter_up = 0

    line_side_prev = {}

    # counters for dwell, ptp, etc.
    movement_threshold = 2
    stationary_history = {}
    queue_length_values, queue_avg_len = [], 20
    queued_ids = set()

    dwell_times = []          # per-object dwell segments while queued
    dwell_start = {}          # timestamp when it became queued

    density_values = {}
    density_flags = {}
    ROI_peak_timers = {}
    peak_time_periods = {}

    density_avg_len = 30

    cv2.namedWindow('SmartFlow')
    calib_lines = load_lines_for_cam(manifest, cam_key)
    if calib_lines:
        print(f"[Lines] Loaded {len(calib_lines)} line(s):",
              [ln["name"] for ln in calib_lines])
    else:
        print("[Lines] No calibrated lines found; a simple center line will be used.")

    # section for --record
    if args.record and source_type in ('ndot', 'video', 'usb'):
        if not resize_dims:
            raise ValueError("Recording requires --resolution.")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        recorder = cv2.VideoWriter('output.avi', fourcc, 30, resize_dims)
    else:
        recorder = None

    # interval set for data
    interval_sec = int(args.interval)
    intersection_id = "INT001"
    last_report_time = time.time()

    # csv logging
    csv_path = "smartflow_metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_header_written = False
    
    # grab a frame
    def read_frame():
        if source_type == 'image_single':
            if not imgs_list: return False, None
            path = imgs_list.pop(0)
            f = cv2.imread(path)
            return True, f
        elif source_type == 'image_folder':
            if not imgs_list: return False, None
            path = imgs_list.pop(0)
            f = cv2.imread(path)
            return True, f
        elif source_type in ('video', 'usb', 'ndot'):
            ok, f = cap.read()
            return ok, f
        elif source_type == 'picamera':
            f = cap.capture_array()
            return True, f
        else:
            return False, None

    roi_masks = None
    roi_areas = None
    roi_names = None
    queue_roi_names = None
    queue_roi_masks = None
    ignore_roi_masks = None

    while True:
        t0 = time.perf_counter()
        ok, frame = read_frame()
        if not ok or frame is None:
            time.sleep(0.05)
            # allow breaking on end of image sequence
            if source_type in ('image_single', 'image_folder'):
                break
            continue

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # init line at mid-height once we know frame size
        height, width = frame.shape[:2]
        # if no calibrated lines yet, define a default horizontal fallback
        if not calib_lines:
            calib_lines = [{
                "name": "L1",
                "p1": (0, height // 2),
                "p2": (width, height // 2),
            }]

        # top-down warp if homography available
        display_frame = frame
        inference_frame = frame
        if use_homography and H is not None and warp_size is not None:
            warp_w, warp_h = warp_size
            inference_frame = cv2.warpPerspective(frame, H, (warp_w, warp_h),
                                                  flags=cv2.INTER_NEAREST,
                                                  borderMode=cv2.BORDER_CONSTANT)
            #display both the original and warped view
            display_frame = cv2.hconcat([
                cv2.resize(frame, (warp_w, warp_h)),
                inference_frame
            ])

            # adjust geometry references to the right half
            height, width = inference_frame.shape[:2]
            line_offset_x = warp_w  # counting line drawn on right image
            draw_target = display_frame
        else:
            draw_target = frame
            line_offset_x = 0  # no offset
            
        # decide scale (meters-per-pixel) for this analysis frame
        # if homography is used this scale applies to the warped frame o/w, it applies to the original
        scale_m_per_px = compute_scale_m_per_px(args, manifest, cam_key, (height, width))

        # ROI setup
        if roi_masks is None:
            roi_masks = {}
            roi_areas = {}
            roi_coords = {}   # keep names for HUD order
           

            # use manifest polygons if requested and it exists
            manifest_rois = manifest.get(cam_key, {}).get("rois", None)
            using_polys = (args.roi_mode == 'manifest') and bool(manifest_rois)

            if using_polys:
                # use polygons as stored from manifest
                masks, areas = build_roi_masks((height, width), manifest_rois)
                if masks:
                    roi_masks = masks
                    roi_areas = areas
                    roi_coords = {name: None for name in roi_masks.keys()}  # compat
                else:
                    print("[ROI] manifest polygons produced no valid masks, falling back to halves.")
                    using_polys = False

            if not using_polys:
                # fallback halves
                roi_coords = {
                    "north": (0, 0, width, height // 2),
                    "south": (0, height // 2, width, height)
                }
                # build masks from these rects
                for name, (x1,y1,x2,y2) in roi_coords.items():
                    poly = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                    m = polygon_mask((height, width), poly)
                    roi_masks[name] = m
                    roi_areas[name] = int(m.sum())
                    
            roi_names = list(roi_masks.keys())

            # init density history and PTP state
            density_values = {name: [] for name in roi_masks.keys()}
            density_flags  = {name: 0 for name in roi_masks.keys()}
            ROI_peak_timers = {name: 0.0 for name in roi_masks.keys()}
            peak_time_periods = {name: [] for name in roi_masks.keys()}

            # decide which ROIs count as queue approaches
            if args.queue_rois:
                requested = [s.strip() for s in args.queue_rois.split(',') if s.strip()]
                # keep only those that actually exist
                queue_roi_names = [name for name in requested if name in roi_names]
                if not queue_roi_names:
                    print(f"[Queue ROI] Requested names {requested} not found; using all ROIs instead.")
                    queue_roi_names = roi_names.copy()
            else:
                # default: all ROIs are queue-eligible
                queue_roi_names = roi_names.copy()

            queue_roi_masks = {name: roi_masks[name] for name in queue_roi_names}
            print(f"[Queue ROI] Using ROIs for queue: {queue_roi_names}")
            
            # decide which ROIs should be ignored
            ignore_roi_masks = None
            if args.ignore_rois:
                requested_ign = [s.strip() for s in args.ignore_rois.split(',') if s.strip()]
                valid_ign = [name for name in requested_ign if name in roi_names]
                if valid_ign:
                    ignore_roi_masks = {name: roi_masks[name] for name in valid_ign}
                    print(f"[Ignore ROI] Ignoring detections in: {valid_ign}")
                else:
                    print(f"[Ignore ROI] None of {requested_ign} matched existing ROIs; ignoring nothing.")
            

        # night-time enhancement
        enhanced_frame = inference_frame
        if args.night_mode:
            # stretch luminance
            ycrcb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            # Contrast Limited Adaptive Histogram Equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y_eq = clahe.apply(y)

            ycrcb_eq = cv2.merge((y_eq, cr, cb))
            enhanced_frame = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

        # YOLO enhanced_frame for night time
        results = model(enhanced_frame, verbose=False)
        dets = results[0].boxes


        # draw all calibrated (or fallback) counting lines
        for ln in calib_lines:
            (x1, y1) = ln["p1"]
            (x2, y2) = ln["p2"]
            # if homography is enabled, inference + metrics are on the right half,
            # so shift x-coords by line_offset_x; y-coords are unchanged
            p1_draw = (line_offset_x + int(x1), int(y1))
            p2_draw = (line_offset_x + int(x2), int(y2))
            cv2.line(draw_target, p1_draw, p2_draw, (0, 255, 255), 2)
            cv2.putText(draw_target, ln["name"],
                        (p1_draw[0] + 5, p1_draw[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        centroids = []
        bboxes = []
        classes = []
        for det in dets:
            xyxy = det.xyxy.cpu().numpy()
            if xyxy.size == 0:
                continue
            xmin, ymin, xmax, ymax = xyxy.astype(int).squeeze()
            conf = float(det.conf.item())
            cls_id = int(det.cls.item())
            classname = labels[cls_id]

            if conf <= min_thresh:
                continue

            if classname not in [
                'articulated_truck',
                'bicycle',
                'bus',
                'car',
                'motorcycle',
                'motorized_vehicle',
                'non-motorized_vehicle',
                'pedestrian',
                'pickup_truck',
                'single_unit_truck',
                'work_van'
            ]:
                continue

            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            # drop detections inside ignore ROIs
            if ignore_roi_masks is not None:
                if centroid_in_any_roi(cx, cy, ignore_roi_masks):
                    # ignore vehicle
                    continue

            centroids.append((cx, cy))
            bboxes.append((xmin, ymin, xmax, ymax))
            classes.append(classname)


        # update tracker
        objects = tracker.update(np.array(centroids) if centroids else [])

        # time reference for this frame
        now_t = time.time()

        # clean up vanished objects; if they were queued, finalize dwell
        for obj_id in list(prev_positions.keys()):
            if obj_id not in objects:
                if obj_id in dwell_start:
                    dwell_times.append(now_t - dwell_start[obj_id])
                    dwell_start.pop(obj_id, None)
                prev_positions.pop(obj_id, None)
                stationary_history.pop(obj_id, None)
                queued_ids.discard(obj_id)

        # density per ROI by mask overlap
        # build a union mask of all detected vehicle rectangles on the inference_frame
        veh_rects = bboxes  # already in inference_frame coords
        vehicle_mask = rect_union_mask((height, width), veh_rects)

        for name, m in roi_masks.items():
            area_roi = max(1, roi_areas[name])
            overlap = (vehicle_mask & m).sum()
            density = overlap / float(area_roi)
            dv = density_values[name]
            dv.append(density)
            if len(dv) > density_avg_len:
                dv.pop(0)
        
        # PTP using smoothed density with hysteresis and min duration
        for name in roi_masks.keys():
            if not density_values[name]:
                continue
            current_density = float(np.mean(density_values[name][-min(len(density_values[name]), density_avg_len):]))
            if current_density > THRESHOLD_DENSITY and density_flags[name] == 0:
                ROI_peak_timers[name] = now_t
                density_flags[name] = 1
            elif current_density <= (THRESHOLD_DENSITY - PTP_BUFFER) and density_flags[name] == 1:
                duration = now_t - ROI_peak_timers[name]
                if duration >= float(args.ptp_min_sec):
                    peak_time_periods[name].append(duration)
                density_flags[name] = 0

        # draw bboxes and IDs, counting and speed, queue logic
        # map centroid index back to bbox to draw
        # build map from object id to nearest centroid index this frame
        for obj_id, (cx, cy) in objects.items():
            # skip objects that have disappeared on this frame to avoid ghost IDs
            if tracker.disappeared.get(obj_id, 0) > 0:
                continue

            # draw dot and ID
            cv2.circle(draw_target, (line_offset_x + cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(draw_target, f'ID {obj_id}', (line_offset_x + cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # line crossing detection + speed
            if obj_id in prev_positions:
                prev_cx, prev_cy = prev_positions[obj_id]

                # check crossing relative to each calibrated line
                for li, ln in enumerate(calib_lines):
                    p1 = ln["p1"]
                    p2 = ln["p2"]
                    prev_s = line_side_prev.get((obj_id, li), None)
                    curr_s = signed_side(cx, cy, p1, p2)

                    if prev_s is not None and prev_s * curr_s < 0:
                        # crossed the line between prev and current frame
                        if prev_s < 0 and curr_s > 0 and obj_id not in counted_ids_down:
                            car_counter_down += 1
                            counted_ids_down.add(obj_id)
                        elif prev_s > 0 and curr_s < 0 and obj_id not in counted_ids_up:
                            car_counter_up += 1
                            counted_ids_up.add(obj_id)

                    line_side_prev[(obj_id, li)] = curr_s

                # speed in px/frame (handle dropped frames)
                dropped = tracker.disappeared.get(obj_id, 0) + 1
                dx = (cx - prev_cx) / dropped
                dy = (cy - prev_cy) / dropped
                ds = math.hypot(dx, dy)
                tracker.speed[obj_id] = ds
                cv2.putText(draw_target, f'v(px/f): {ds:.1f}',
                            (line_offset_x + cx + 12, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                dx = dy = 0.0


            prev_positions[obj_id] = (cx, cy)

            # queue region and dwell logic
            # only consider slow/stopped vehicles whose centroid lies inside ANY ROI
            active_queue_masks = queue_roi_masks if queue_roi_masks else roi_masks
            is_in_queue_area = centroid_in_any_roi(cx, cy, active_queue_masks)
            is_slow = abs(dy) < movement_threshold

            if is_in_queue_area and is_slow:
                stationary_history[obj_id] = stationary_history.get(obj_id, 0) + 1

                # after a few consecutive frames of being slow inside an ROI, mark as queued
                if stationary_history[obj_id] > 15:
                    if obj_id not in queued_ids:
                        queued_ids.add(obj_id)
                        # start dwell timer when it becomes queued
                        dwell_start[obj_id] = now_t
            else:
                # if it was queued and now left the ROI or sped up, finalize dwell
                if obj_id in queued_ids:
                    queued_ids.discard(obj_id)
                    if obj_id in dwell_start:
                        dwell_times.append(now_t - dwell_start[obj_id])
                        dwell_start.pop(obj_id, None)
                stationary_history[obj_id] = 0

            # only show dwell for actually queued vehicles
            if obj_id in dwell_start:
                elapsed = now_t - dwell_start[obj_id]
                cv2.putText(draw_target, f'Dwell {elapsed:.1f}s',
                            (line_offset_x + cx - 10, cy + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # queue length & smoothing (vehicles)
        queue_length = len(queued_ids)
        queue_length_values.append(queue_length)
        if len(queue_length_values) > queue_avg_len:
            queue_length_values.pop(0)
        queue_length_smoothed = float(np.mean(queue_length_values))

        # per-ROI queue stats + queue length in meters
        # use queue-eligible ROIs if specified, else all ROIs
        active_queue_masks = queue_roi_masks if queue_roi_masks else roi_masks
        active_queue_names = list(active_queue_masks.keys())

        # initialize per-ROI structures
        queue_count_per_roi = {name: 0 for name in active_queue_names}
        queue_len_m_per_roi = {name: None for name in active_queue_names}
        queue_len_m = None

        if queued_ids:
            # group queued centroids by ROI
            roi_to_points = {name: [] for name in active_queue_names}
            for qid in list(queued_ids):
                if qid in objects:
                    cx, cy = objects[qid]
                    roi_name = which_roi_contains(cx, cy, active_queue_masks)
                    if roi_name is not None:
                        roi_to_points[roi_name].append((cx, cy))

            # fill per-ROI queue counts
            for name, pts in roi_to_points.items():
                queue_count_per_roi[name] = len(pts)

            # compute per-ROI and global queue length in meters if scale is known
            if scale_m_per_px is not None:
                max_span_px = 0.0
                for name, pts in roi_to_points.items():
                    if len(pts) < 2:
                        continue  # need at least two vehicles for a span
                    local_max = 0.0
                    for i in range(len(pts)):
                        x1, y1 = pts[i]
                        for j in range(i + 1, len(pts)):
                            x2, y2 = pts[j]
                            d = math.hypot(x2 - x1, y2 - y1)
                            if d > local_max:
                                local_max = d
                    if local_max > 0:
                        span_m = local_max * scale_m_per_px
                        queue_len_m_per_roi[name] = span_m
                        if local_max > max_span_px:
                            max_span_px = local_max

                if max_span_px > 0:
                    queue_len_m = max_span_px * scale_m_per_px

        y0 = 100
        
        queue_text = f'Queue Len: {queue_length_smoothed:.1f} veh'
        if queue_len_m is not None:
            queue_text += f'  ({queue_len_m:.1f} m)'
        cv2.putText(draw_target, queue_text, (10, y0 + 30 * len(roi_names)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # HUD
        cv2.putText(draw_target, f'Down: {car_counter_down}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(draw_target, f'Up:   {car_counter_up}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for idx, name in enumerate(roi_names):
            sm = float(np.mean(density_values[name])) if density_values[name] else 0.0
            cv2.putText(draw_target, f'{name} density: {sm:.2f}', (10, y0 + 30 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS
        t1 = time.perf_counter()
        fps = 1.0 / max(1e-6, (t1 - t0))
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_fps = float(np.mean(frame_rate_buffer))
        
        # draw FPS in top-right
        fps_text = f'FPS: {avg_fps:.1f}'
        (h_fps, w_fps), baseline = cv2.getTextSize(
            fps_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        h_img, w_img = draw_target.shape[:2]
        x = w_img - w_fps - 100
        y = 30  # 10 px margin from right, a bit down from the top

        cv2.putText(draw_target, fps_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        
        #convert if scale is avaiable for vehicle speed
        avg_speed_mps = None
        avg_speed_mph = None
        if scale_m_per_px is not None and avg_fps > 0:
            # convert each tracked speed to m/s
            # take mean over currently visible IDs
            speeds_mps = []
            for oid, px_per_frame in tracker.speed.items():
                mps = px_per_frame * avg_fps * scale_m_per_px
                speeds_mps.append(mps)
            if speeds_mps:
                avg_speed_mps = float(np.mean(speeds_mps))
                avg_speed_mph = avg_speed_mps * 2.23693629
        
        #and if avail, post it on screen        
        if avg_speed_mph is not None:
            cv2.putText(draw_target, f'Avg speed: {avg_speed_mph:.1f} mph', (10, y0 + 30 * (len(roi_names)+1)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
        # flow per hour based on interval counters
        flow_per_hr = (car_counter_up + car_counter_down) * (3600.0 / max(1, interval_sec))

        # dwell avg for IDs that exited this frame
        avg_dwell = float(np.mean(dwell_times)) if dwell_times else 0.0

        # report
        now = time.time()
        if now - last_report_time >= interval_sec:
            total = car_counter_down + car_counter_up
            
            # prepare metric dicts for clarity
            metrics_density = {
                name: round(float(np.mean(density_values[name])), 3) if density_values[name] else 0.0
                for name in roi_names
            }

            # queue metrics: use the same active queue ROI names we built earlier
            active_queue_names = list(
                (queue_roi_masks.keys() if queue_roi_masks else roi_masks.keys())
            )

            metrics_queue_len_m = round(queue_len_m, 2) if queue_len_m is not None else None
            metrics_queue_len_m_per_roi = {
                name: (round(queue_len_m_per_roi.get(name), 2)
                       if queue_len_m_per_roi.get(name) is not None else None)
                for name in active_queue_names
            }
            metrics_queue_count_per_roi = {
                name: int(queue_count_per_roi.get(name, 0))
                for name in active_queue_names
            }

            metrics_ptp_last_counts = {
                name: len(peak_time_periods[name]) for name in roi_names
            }
            
            report = {
                "type": "traffic_report",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "intersection_id": intersection_id,
                "interval_sec": interval_sec,
                "vehicle_counts": total,
                "directions": {"down": car_counter_down, "up": car_counter_up},
                "metrics": {
                    "density": metrics_density,
                    "queue_length_veh": round(queue_length_smoothed, 2),
                    "queue_length_m": metrics_queue_len_m,
                    "queue_length_veh_per_roi": metrics_queue_count_per_roi,
                    "queue_length_m_per_roi": metrics_queue_len_m_per_roi,
                    "flow": round(flow_per_hr, 2),
                    "dwell": round(avg_dwell, 2),
                    "avg_speed_mph": round(avg_speed_mph, 2) if avg_speed_mph is not None else None,
                    "ptp_last_counts": metrics_ptp_last_counts,
                }
            }
            try:
                if ser:
                    ser.write((json.dumps(report) + "\n").encode('utf-8'))
                print("[UART/LOG] Sent:", report)
            except Exception as e:
                print("[UART] Error:", e)


            # csv logging
            # write a header row the first time its generated
            # log the same metrics we are sending in uart
            # one-time header with ROI density columns
            if not csv_header_written:
                header = [
                    "timestamp",
                    "intersection_id",
                    "interval_sec",
                    "total_vehicles",
                    "down_count",
                    "up_count",
                    "queue_length_veh",
                    "queue_len_m",
                    "flow_veh_per_hr",
                    "avg_dwell_s",
                    "avg_speed_mph",
                ]
                for name in roi_names:
                    header.append(f"density_{name}")
                csv_writer.writerow(header)
                csv_header_written = True

            row = [
                report["timestamp"],
                report["intersection_id"],
                report["interval_sec"],
                total,
                report["directions"]["down"],
                report["directions"]["up"],
                queue_length_smoothed,
                queue_len_m if queue_len_m is not None else "",
                flow_per_hr,
                avg_dwell,
                avg_speed_mph if avg_speed_mph is not None else "",
            ]
            for name in roi_names:
                row.append(report["metrics"]["density"][name])

            csv_writer.writerow(row)
            csv_file.flush()
            
            # reset counters for next interval
            car_counter_down = 0
            car_counter_up = 0
            counted_ids_down.clear()
            counted_ids_up.clear()
            dwell_times.clear()
            last_report_time = now
            
        # show
        cv2.imshow('SmartFlow', draw_target)
        if recorder is not None:
            # if showing side-by-side due to homography, recorder expects resize_dims
            if draw_target.shape[1] != (resize_dims[0] if resize_dims else draw_target.shape[1]) or \
               draw_target.shape[0] != (resize_dims[1] if resize_dims else draw_target.shape[0]):
                rec_frame = cv2.resize(draw_target, resize_dims)
            else:
                rec_frame = draw_target
            recorder.write(rec_frame)

        # quit 'q'
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q')):
            break

    # cleanup
    if source_type in ('video', 'usb', 'ndot'):
        cap.release()
    if recorder is not None:
        recorder.release()
    # csv logging
    if 'csv_file' in locals():
        csv_file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
