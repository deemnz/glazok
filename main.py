# -*- coding: utf-8 -*-
import os
import time
import json
import cv2
import av
import torch
import datetime
import numpy as np
import webbrowser
import subprocess
import logging
import math
from ultralytics import YOLO
from scipy.spatial import distance
from db import init_db, upsert_session, get_all_sessions

# ---------------------------
# Logging configuration
# ---------------------------
logging.basicConfig(filename="debug.log", level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s: %(message)s")

# ---------------------------
# Global configuration file and defaults
# ---------------------------
CONFIG_FILE = "launch_config.json"
default_config = {
    "rtsp_url": "",
    "object_type": "car",
    "analysis_mode": "unique",  # "directional" or "unique"
    "line_options": {
        "orientation": "horizontal",  # "horizontal", "vertical" or "diagonal"
        "position": 0.5,
        "direction_mode": "vertical"   # horizontal: vertical; vertical: horizontal; diagonal: "diag1" or "diag2"
    },
    "resolution_width": 640,
    "resolution_height": 360,
    "counting_algorithm": "standard",  # "standard" or "threshold"
    "min_displacement": 10,            # in pixels
    "record_interval": 60              # record save time, seconds
}

flask_process = None

# ---------------------------
# Load / Save launch settings
# ---------------------------
def load_launch_settings():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
                return config
            except Exception as e:
                logging.error("Error loading config: %s", e)
                return default_config.copy()
    else:
        return default_config.copy()

def save_launch_settings(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

# ---------------------------
# Get distinct RTSP URLs from DB
# ---------------------------
def get_distinct_rtsp_urls():
    from db import get_all_sessions
    sessions = get_all_sessions()
    urls = list({session["stream_url"] for session in sessions})
    return urls

# ---------------------------
# Centroid Tracker (simple version)
# ---------------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.previousCentroids = {}
        self.counted = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.previousCentroids[self.nextObjectID] = centroid
        self.counted[self.nextObjectID] = False
        self.nextObjectID += 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
            del self.disappeared[objectID]
            del self.previousCentroids[objectID]
            del self.counted[objectID]

    def update(self, inputCentroids):
        if len(inputCentroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = distance.cdist(np.array(objectCentroids), np.array(inputCentroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.previousCentroids[objectID] = self.objects[objectID]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for col in unusedCols:
                self.register(inputCentroids[col])
        return self.objects

# ---------------------------
# Functions for handling RTSP URLs (choose from DB or input new)
# ---------------------------
def choose_rtsp_url(config):
    existing_links = get_distinct_rtsp_urls()
    if existing_links:
        print("\nExisting RTSP links from database:")
        for idx, link in enumerate(existing_links):
            print(f"{idx+1}. {link}")
        choice = input("Select a link by number or press Enter to keep current/new URL: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(existing_links):
                config["rtsp_url"] = existing_links[idx]
                logging.debug("Selected RTSP URL from DB: %s", config["rtsp_url"])
                return config["rtsp_url"]
    if config["rtsp_url"]:
        print("\nCurrent RTSP URL: {}".format(config["rtsp_url"]))
        new_link = input("Press Enter to use it or type a new URL: ").strip()
        if new_link != "":
            config["rtsp_url"] = new_link
    else:
        config["rtsp_url"] = input("Enter RTSP URL: ").strip()
    return config["rtsp_url"]

# ---------------------------
# Function to choose object type (using saved settings)
# ---------------------------
def choose_object_type_setting(config, model):
    if hasattr(model.model, 'names'):
        names = model.model.names
        print("\nAvailable object types:")
        for k, v in sorted(names.items()):
            print(f"{k}: {v}")
    print("\nCurrent object type: {}".format(config["object_type"]))
    return config["object_type"]

# ---------------------------
# Function to choose analysis mode
# ---------------------------
def choose_analysis_mode_setting(config):
    print("\nSelect analysis mode:")
    print("1. Directional counting (with crossing line)")
    print("2. Unique counting (count unique objects in frame)")
    choice = input("Your choice (1/2) [current: {}]: ".format(config["analysis_mode"])).strip()
    if choice == "1":
        config["analysis_mode"] = "directional"
    elif choice == "2":
        config["analysis_mode"] = "unique"
    return config["analysis_mode"]

# ---------------------------
# Function to choose counting algorithm variant
# ---------------------------
def choose_counting_algorithm_setting(config):
    print("\nSelect counting algorithm variant:")
    print("1. Standard")
    print("2. Threshold-based (min displacement required)")
    choice = input("Your choice (1/2) [current: {}]: ".format(config.get("counting_algorithm", "standard"))).strip()
    if choice == "1":
        config["counting_algorithm"] = "standard"
    elif choice == "2":
        config["counting_algorithm"] = "threshold"
    return config["counting_algorithm"]

# ---------------------------
# Function to choose line options (for directional mode)
# ---------------------------
def choose_line_options_setting(config):
    print("\nSelect crossing line type:")
    print("1. Horizontal (count up/down)")
    print("2. Vertical (count left/right)")
    print("3. Diagonal")
    orientation_choice = input("Your choice (1/2/3) [current: {}]: ".format(config["line_options"]["orientation"])).strip()
    if orientation_choice == "1":
        config["line_options"]["orientation"] = "horizontal"
    elif orientation_choice == "2":
        config["line_options"]["orientation"] = "vertical"
    elif orientation_choice == "3":
        config["line_options"]["orientation"] = "diagonal"
    else:
        print("Invalid input. Keeping current value.")
    pos = input("Enter relative position of the line (0.0 to 1.0) [current: {}]: ".format(config["line_options"]["position"])).strip()
    try:
        if pos != "":
            pos_val = float(pos)
            if 0 <= pos_val <= 1:
                config["line_options"]["position"] = pos_val
    except:
        pass
    if config["line_options"]["orientation"] == "diagonal":
        print("Select diagonal direction:")
        print("1. Top-left to bottom-right")
        print("2. Top-right to bottom-left")
        d_choice = input("Your choice (1/2) [current: {}]: ".format(config["line_options"]["direction_mode"])).strip()
        if d_choice == "1":
            config["line_options"]["direction_mode"] = "diag1"
        elif d_choice == "2":
            config["line_options"]["direction_mode"] = "diag2"
    else:
        config["line_options"]["direction_mode"] = "vertical" if config["line_options"]["orientation"] == "horizontal" else "horizontal"
    return config["line_options"]

# ---------------------------
# Function to change window resolution
# ---------------------------
def change_resolution(config):
    try:
        new_width = int(input("Enter window width (current: {}): ".format(config["resolution_width"])))
        new_height = int(input("Enter window height (current: {}): ".format(config["resolution_height"])))
        config["resolution_width"] = new_width
        config["resolution_height"] = new_height
        print("Resolution updated: {}x{}.".format(new_width, new_height))
    except:
        print("Invalid input. Resolution not changed.")
    return config

# ---------------------------
# Function to change record interval (in seconds)
# ---------------------------
def change_record_interval(config):
    try:
        new_interval = int(input("Enter record interval in seconds (current: {}): ".format(config.get("record_interval", 60))))
        config["record_interval"] = new_interval
        print("Record interval updated to {} seconds.".format(new_interval))
    except:
        print("Invalid input. Record interval not changed.")
    return config

# ---------------------------
# Function to change launch settings
# ---------------------------
def change_launch_settings(config, model):
    print("\n=== Change Launch Settings ===")
    choice = input("Do you want to select an existing RTSP link from the database? (y/n): ").strip().lower()
    if choice == "y":
        existing_links = get_distinct_rtsp_urls()
        if existing_links:
            print("Existing RTSP links:")
            for idx, link in enumerate(existing_links):
                print(f"{idx+1}. {link}")
            selected = input("Select a link by number (or press Enter to skip): ").strip()
            if selected.isdigit():
                idx = int(selected) - 1
                if 0 <= idx < len(existing_links):
                    config["rtsp_url"] = existing_links[idx]
        else:
            print("No RTSP links found in the database.")
            config["rtsp_url"] = input("Enter RTSP URL: ").strip()
    else:
        config["rtsp_url"] = input("Enter RTSP URL (leave empty to keep current [{}]): ".format(config["rtsp_url"])).strip() or config["rtsp_url"]

    config["object_type"] = input("Enter object type to count (e.g. person, car) (leave empty to keep current [{}]): ".format(config["object_type"])).strip() or config["object_type"]
    choose_analysis_mode_setting(config)
    choose_counting_algorithm_setting(config)
    if config["analysis_mode"] == "directional":
        choose_line_options_setting(config)
    change_resolution(config)
    change_record_interval(config)
    if config.get("counting_algorithm", "standard") == "threshold":
        explanation = ("Threshold-based algorithm explanation:\n"
                       "An object is only counted if the difference between its current and previous positions\n"
                       "exceeds the threshold (min_displacement). This helps filter out small movements due\n"
                       "to noise or jitter, preventing multiple counts of the same object.")
        print(explanation)
        try:
            new_thresh = input("Enter minimum displacement threshold in pixels (current: {}): ".format(config["min_displacement"])).strip()
            if new_thresh != "":
                config["min_displacement"] = int(new_thresh)
        except:
            print("Invalid input. Keeping current threshold.")
    save_launch_settings(config)
    print("Launch settings saved.")
    logging.debug("Launch settings updated: %s", config)
    return config

# ---------------------------
# STREAM PROCESSING FUNCTIONS (with periodic DB upsert)
# ---------------------------
# Directional counting mode
def run_stream_directional(rtsp_url, object_type, line_options, model, resolution, counting_algorithm, min_disp, record_interval):
    logging.debug("Starting directional stream processing with rtsp_url=%s, object_type=%s", rtsp_url, object_type)
    try:
        container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
    except av.AVError as e:
        logging.error("Error opening RTSP stream: %s", e)
        print("Error opening RTSP stream:", e)
        return (0,0), 0
    ct = CentroidTracker(maxDisappeared=40)
    if line_options["orientation"] == "horizontal":
        up_count = 0
        down_count = 0
    elif line_options["orientation"] == "vertical":
        left_count = 0
        right_count = 0
    elif line_options["orientation"] == "diagonal":
        diag1_count = 0
        diag2_count = 0
    last_record_time = time.time()
    session_start = datetime.datetime.now()  # start time
    logging.debug("Directional session started at %s", session_start)
    while True:
        try:
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="bgr24")
                (H, W) = img.shape[:2]
                if line_options["orientation"] == "horizontal":
                    line_coord = int(H * line_options["position"])
                    pt1, pt2 = (0, line_coord), (W, line_coord)
                elif line_options["orientation"] == "vertical":
                    line_coord = int(W * line_options["position"])
                    pt1, pt2 = (line_coord, 0), (line_coord, H)
                elif line_options["orientation"] == "diagonal":
                    if line_options["direction_mode"] == "diag1":
                        offset = int(min(W, H) * line_options["position"])
                        pt1, pt2 = (0, offset), (W, H - offset)
                    else:
                        offset = int(min(W, H) * line_options["position"])
                        pt1, pt2 = (W, offset), (0, H - offset)
                else:
                    pt1, pt2 = (0, int(H/2)), (W, int(H/2))
                results = model(img)
                current_centroids = []
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        xyxy = box.xyxy.cpu().numpy()[0]
                        x1, y1, x2, y2 = map(int, xyxy)
                        cls = int(box.cls.cpu().numpy()[0])
                        label = model.model.names[cls] if hasattr(model.model, 'names') else str(cls)
                        if label.lower() == object_type.lower():
                            cX = int((x1 + x2) / 2)
                            cY = int((y1 + y2) / 2)
                            current_centroids.append((cX, cY))
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)
                            cv2.putText(img, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                objects = ct.update(current_centroids)
                for (objectID, centroid) in objects.items():
                    prev = ct.previousCentroids.get(objectID, centroid)
                    cv2.putText(img, f"ID {objectID}", (centroid[0]-10, centroid[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)
                    if not ct.counted[objectID]:
                        crossed = False
                        # Horizontal: compare Y values
                        if line_options["orientation"] == "horizontal":
                            if prev[1] > pt1[1] and centroid[1] <= pt1[1]:
                                up = True
                                crossed = True
                            elif prev[1] < pt1[1] and centroid[1] >= pt1[1]:
                                up = False
                                crossed = True
                        # Vertical: compare X values
                        elif line_options["orientation"] == "vertical":
                            if prev[0] > pt1[0] and centroid[0] <= pt1[0]:
                                left = True
                                crossed = True
                            elif prev[0] < pt1[0] and centroid[0] >= pt1[0]:
                                left = False
                                crossed = True
                        # Diagonal: compute length
                        elif line_options["orientation"] == "diagonal":
                            vx = pt2[0] - pt1[0]
                            vy = pt2[1] - pt1[1]
                            norm = math.hypot(vx, vy)
                            if norm == 0:
                                norm = 1
                            d_prev = ((prev[0] - pt1[0]) * vy - (prev[1] - pt1[1]) * vx) / norm
                            d_curr = ((centroid[0] - pt1[0]) * vy - (centroid[1] - pt1[1]) * vx) / norm
                            if d_prev * d_curr < 0:
                                crossed = True
                                diag = (d_prev > 0)
                        if crossed:
                            if line_options["orientation"] in ["horizontal", "vertical"]:
                                if line_options["orientation"] == "horizontal":
                                    disp = abs(centroid[1] - prev[1])
                                else:
                                    disp = abs(centroid[0] - prev[0])
                            else:
                                disp = abs((centroid[0] + centroid[1]) - (prev[0] + prev[1]))
                            if counting_algorithm == "threshold" and disp < min_disp:
                                logging.debug("Object ID %s displacement %s below threshold %s; not counted", objectID, disp, min_disp)
                            else:
                                if line_options["orientation"] == "horizontal":
                                    if prev[1] > pt1[1] and centroid[1] <= pt1[1]:
                                        up_count += 1
                                        logging.debug("Object ID %s counted as UP", objectID)
                                    elif prev[1] < pt1[1] and centroid[1] >= pt1[1]:
                                        down_count += 1
                                        logging.debug("Object ID %s counted as DOWN", objectID)
                                elif line_options["orientation"] == "vertical":
                                    if prev[0] > pt1[0] and centroid[0] <= pt1[0]:
                                        left_count += 1
                                        logging.debug("Object ID %s counted as LEFT", objectID)
                                    elif prev[0] < pt1[0] and centroid[0] >= pt1[0]:
                                        right_count += 1
                                        logging.debug("Object ID %s counted as RIGHT", objectID)
                                elif line_options["orientation"] == "diagonal":
                                    if diag:
                                        diag1_count += 1
                                        logging.debug("Object ID %s counted as Diagonal1", objectID)
                                    else:
                                        diag2_count += 1
                                        logging.debug("Object ID %s counted as Diagonal2", objectID)
                                ct.counted[objectID] = True
                cv2.line(img, pt1, pt2, (255, 0, 0), 2)
                if line_options["orientation"] == "horizontal":
                    current_total = up_count + down_count
                    direction_text = f"Up: {up_count}   Down: {down_count}"
                elif line_options["orientation"] == "vertical":
                    current_total = left_count + right_count
                    direction_text = f"Left: {left_count}   Right: {right_count}"
                elif line_options["orientation"] == "diagonal":
                    current_total = diag1_count + diag2_count
                    direction_text = f"Diagonal1: {diag1_count}   Diagonal2: {diag2_count}"
                else:
                    current_total = 0
                    direction_text = ""
                elapsed_str = session_start.strftime("%d-%m-%Y %H:%M:%S")
                counter_text = f"Total {object_type}: {current_total}   ({direction_text})   since {elapsed_str}"
                cv2.putText(img, counter_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                display_img = cv2.resize(img, (resolution["width"], resolution["height"]))
                cv2.imshow("RTSP - Directional Counting", display_img)
                
                now = time.time()
                if now - last_record_time >= record_interval:
                    record_start = datetime.datetime.fromtimestamp(last_record_time)
                    record_end = datetime.datetime.now()
                    delta_total = current_total
                    if line_options["orientation"] == "horizontal":
                        delta_dir1 = up_count
                        delta_dir2 = down_count
                    elif line_options["orientation"] == "vertical":
                        delta_dir1 = left_count
                        delta_dir2 = right_count
                    elif line_options["orientation"] == "diagonal":
                        delta_dir1 = diag1_count
                        delta_dir2 = diag2_count
                    else:
                        delta_dir1 = delta_dir2 = 0
                    try:
                        upsert_session(rtsp_url, object_type, delta_dir1, delta_dir2, delta_total,
                                       session_start.strftime("%d-%m-%Y %H:%M:%S"),
                                       record_end.strftime("%d-%m-%Y %H:%M:%S"))
                        logging.debug("Recorded interval: %s - %s, total=%s", record_start, record_end, delta_total)
                        print(f"Recorded interval: {record_start.strftime('%H:%M:%S')} - {record_end.strftime('%H:%M:%S')}, total: {delta_total}")
                    except Exception as e:
                        logging.error("Error inserting interval into DB: %s", e)
                    last_record_time = now
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    container.close()
                    cv2.destroyAllWindows()
                    logging.debug("Directional counting session ended with total=%s", current_total)
                    if line_options["orientation"] == "horizontal":
                        return (up_count, down_count), current_total
                    elif line_options["orientation"] == "vertical":
                        return (left_count, right_count), current_total
                    elif line_options["orientation"] == "diagonal":
                        return ((diag1_count, diag2_count), current_total)
                    else:
                        return (0,0), current_total
        except Exception as e:
            logging.error("Error during directional stream processing: %s", e)
            print("Error during stream processing:", e)
            break
    container.close()
    cv2.destroyAllWindows()
    return (0,0), 0

# Unique counting mode
def run_stream_unique(rtsp_url, object_type, model, resolution, record_interval):
    logging.debug("Starting unique stream processing with rtsp_url=%s, object_type=%s", rtsp_url, object_type)
    try:
        container = av.open(rtsp_url, options={'rtsp_transport': 'tcp'})
    except av.AVError as e:
        logging.error("Error opening RTSP stream: %s", e)
        print("Error opening RTSP stream:", e)
        return 0
    ct = CentroidTracker(maxDisappeared=40)
    total_count = 0
    last_record_time = time.time()
    session_start = datetime.datetime.now()
    logging.debug("Unique session started at %s", session_start)
    while True:
        try:
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="bgr24")
                (H, W) = img.shape[:2]
                results = model(img)
                current_centroids = []
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        xyxy = box.xyxy.cpu().numpy()[0]
                        x1, y1, x2, y2 = map(int, xyxy)
                        cls = int(box.cls.cpu().numpy()[0])
                        label = model.model.names[cls] if hasattr(model.model, 'names') else str(cls)
                        if label.lower() == object_type.lower():
                            cX = int((x1 + x2) / 2)
                            cY = int((y1 + y2) / 2)
                            current_centroids.append((cX, cY))
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)
                            cv2.putText(img, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                objects = ct.update(current_centroids)
                for (objectID, centroid) in objects.items():
                    if not ct.counted[objectID]:
                        total_count += 1
                        ct.counted[objectID] = True
                    cv2.putText(img, f"ID {objectID}", (centroid[0]-10, centroid[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)
                elapsed_str = session_start.strftime("%d-%m-%Y %H:%M:%S")
                counter_text = f"Total {object_type}: {total_count} since {elapsed_str}"
                cv2.putText(img, counter_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                display_img = cv2.resize(img, (resolution["width"], resolution["height"]))
                cv2.imshow("RTSP - Unique Counting", display_img)
                now = time.time()
                if now - last_record_time >= record_interval:
                    record_start = datetime.datetime.fromtimestamp(last_record_time)
                    record_end = datetime.datetime.now()
                    try:
                        upsert_session(rtsp_url, object_type, 0, 0, total_count,
                                       record_start.strftime("%d-%m-%Y %H:%M:%S"),
                                       record_end.strftime("%d-%m-%Y %H:%M:%S"))
                        logging.debug("Unique interval recorded: %s - %s, total=%s", record_start, record_end, total_count)
                        print(f"Recorded unique interval: {record_start.strftime('%H:%M:%S')} - {record_end.strftime('%H:%M:%S')}, total: {total_count}")
                    except Exception as e:
                        logging.error("Error inserting unique interval into DB: %s", e)
                    last_record_time = now
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    container.close()
                    cv2.destroyAllWindows()
                    logging.debug("Unique counting session ended with total=%s", total_count)
                    return total_count
        except Exception as e:
            logging.error("Error during unique stream processing: %s", e)
            print("Error during stream processing:", e)
            break
    container.close()
    cv2.destroyAllWindows()
    return total_count

# ---------------------------
# Launch analysis and Flask web interface in parallel
# ---------------------------
def launch_analysis_and_flask(config, model):
    logging.debug("Launching analysis and Flask with config: %s", config)
    print("\nLaunching Flask web analytics interface...")
    flask_proc = subprocess.Popen(["python", "webapp.py"])
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5000")
    
    rtsp_url = config["rtsp_url"]
    object_type = config["object_type"]
    analysis_mode = config["analysis_mode"]
    resolution = {"width": config["resolution_width"], "height": config["resolution_height"]}
    counting_algorithm = config["counting_algorithm"]
    min_disp = config["min_displacement"]
    record_interval = config.get("record_interval", 60)
    
    session_start = datetime.datetime.now()
    print(f"\nSession started: {session_start.strftime('%d-%m-%Y %H:%M:%S')}")
    logging.debug("Session started at %s", session_start)
    
    if analysis_mode == "directional":
        ret = run_stream_directional(rtsp_url, object_type, config["line_options"], model, resolution, counting_algorithm, min_disp, record_interval)
        if config["line_options"]["orientation"] == "horizontal":
            (up_count, down_count), total = ret
            direction_info = f"Up: {up_count}, Down: {down_count}"
        elif config["line_options"]["orientation"] == "vertical":
            (left_count, right_count), total = ret
            direction_info = f"Left: {left_count}, Right: {right_count}"
        elif config["line_options"]["orientation"] == "diagonal":
            ((diag1, diag2), total) = ret
            direction_info = f"Diagonal1: {diag1}, Diagonal2: {diag2}"
        else:
            total = 0
            direction_info = ""
    else:
        total = run_stream_unique(rtsp_url, object_type, model, resolution, record_interval)
        direction_info = ""
    
    session_end = datetime.datetime.now()
    print(f"Session ended: {session_end.strftime('%d-%m-%Y %H:%M:%S')}")
    if analysis_mode == "directional":
        print(f"Counted {total} objects ({direction_info})")
    else:
        print(f"Counted {total} objects")
    
    logging.debug("Session ended at %s with total count: %s", session_end, total)
    try:
        upsert_session(rtsp_url, object_type, 
                       (ret[0][0] if analysis_mode=="directional" and config["line_options"]["orientation"] in ["horizontal", "vertical"] else 0),
                       (ret[0][1] if analysis_mode=="directional" and config["line_options"]["orientation"] in ["horizontal", "vertical"] else 0),
                       total,
                       session_start.strftime("%d-%m-%Y %H:%M:%S"),
                       session_end.strftime("%d-%m-%Y %H:%M:%S"))
        logging.debug("Session upserted into DB successfully.")
        print("Session data upserted into DB successfully.")
    except Exception as e:
        logging.error("Error upserting session into DB: %s", e)
        print("Error upserting session into DB:", e)
    
    flask_proc.terminate()

# ---------------------------
# MAIN MENU
# ---------------------------
def main_menu():
    init_db()
    config = load_launch_settings()
    config = {**default_config, **config}
    
    model = YOLO("yolo11n.pt")
    if torch.cuda.is_available():
        device = "cuda:0"
        model.model = model.model.to(device)
        print("Using GPU:", device)
    else:
        print("GPU not found. Using CPU.")
        
    while True:
        print("\n=== Main Menu ===")
        print("1. Launch analysis with saved settings")
        print("2. Change launch settings")
        print("3. View analytics (console)")
        print("4. Open web analytics in browser")
        print("5. Exit")
        choice = input("Your choice: ").strip()
        if choice == "1":
            if config["rtsp_url"] == "":
                config["rtsp_url"] = input("Enter RTSP URL: ").strip()
                save_launch_settings(config)
            logging.debug("Launching analysis with saved settings: %s", config)
            launch_analysis_and_flask(config, model)
        elif choice == "2":
            config = change_launch_settings(config, model)
        elif choice == "3":
            sessions = get_all_sessions()
            print("\nAnalytics:")
            for row in sessions:
                # No ID column
                print(f"Stream: {row['stream_url']}, Object: {row['object_type']}, Total: {row['total']}, Session Start: {row['session_start']}, Session End: {row['session_end']}")
        elif choice == "4":
            print("\nOpening web analytics in browser...")
            webbrowser.open("http://127.0.0.1:5000")
        elif choice == "5":
            print("Exiting.")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main_menu()
