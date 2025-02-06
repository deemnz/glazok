# Glazok

Glazok is a video stream analytics system built in Python that processes RTSP streams. It uses a YOLO model to detect objects in real time, counts objects in two modes (directional and unique counting), and periodically records analytics data in a SQLite database. A Flask-based web interface is provided for viewing the analytics.

## Features

- **RTSP Stream Support with Digital Authentication**  
  Supports RTSP streams (e.g., `rtsp://admin:password@ip:port/...`).

- **Object Detection Using YOLO**  
  Uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (e.g., `yolo11n.pt`) for object detection.

- **Two Counting Modes**  
  - **Unique Counting:** Counts unique objects detected in the frame.
  - **Directional Counting:** Counts objects that cross a user-defined line.  
    Users can choose the line orientation (horizontal, vertical, or diagonal).  
    For diagonal lines, the signed distance from the object’s previous and current positions to the line is calculated. A crossing is detected when the sign of the distance changes.

- **Threshold-based Counting (Optional)**  
  When the "threshold" variant is selected, an object is only counted if the difference between its current and previous positions (along the chosen axis or the sum for diagonal lines) is greater than or equal to the specified `min_displacement`. This helps to filter out minor movements caused by noise or camera jitter, preventing multiple counts of the same object.

- **Real-time Analytics Recording**  
  Analytics data is updated and recorded in the database every `record_interval` seconds (set in the configuration).  
  The database table uses a composite primary key (`stream_url`, `session_start`), allowing multiple sessions for the same stream (if the analysis starts at different times). If a record for the same stream and session start already exists, it is updated.

- **Configuration via JSON**  
  All settings (RTSP URL, object type, analysis mode, counting algorithm, threshold, line parameters, window resolution, and record interval) are stored in `launch_config.json`.

- **Web Analytics Interface**  
  A Flask-based web interface is provided:
  - The **Overview** page (`index.html`) displays a list of streams grouped by RTSP URL, with the number of recorded sessions and the start date.
  - A **Stream Details** page shows detailed analytics for each stream.
  - A **Diagram** page allows the user to select a date (via a dropdown list or manual entry) to visualize the data using Chart.js. The page refreshes automatically (default every 60 seconds).

## Project Structure

- **db.py**  
  Contains functions for working with the SQLite database. The table is created with a composite primary key (`stream_url`, `session_start`), and the `upsert_session` function performs an upsert operation.

- **main.py**  
  The main application for configuring and launching the stream analysis. It includes:
  - Functions for loading and saving settings from/to `launch_config.json`.
  - User menus to change settings (selecting an RTSP URL from the database or entering a new one, choosing object type from available options in the YOLO model, selecting the analysis mode, counting algorithm, threshold, record interval, and line parameters).
  - Real-time stream processing functions that update analytics every `record_interval` seconds and perform an upsert into the database.
  - In directional mode, if a diagonal line is selected, a signed distance method is used to detect crossings.

- **webapp.py**  
  A Flask application that provides the analytics web interface. It includes routes for:
  - `/` – Overview page listing streams with session counts.
  - `/stream/<stream_url>` – Detailed analytics for a given stream.
  - `/diagram` – A diagram page with date filtering (via dropdown and manual input) that displays a Chart.js line chart.
  - `/data` – An API endpoint to return analytics data in JSON format.

- **templates/**  
  Contains HTML templates:
  - **index.html** – Overview of streams using a Bootstrap accordion.
  - **stream_details.html** – Detailed analytics for a specific stream (without the ID column).
  - **diagram.html** – A page for visualizing the analytics using Chart.js, with date selection.

## Installation and Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/deemnz/glazok.git
   cd glazok

2. **Install dependencies:**

The project requires:

- OpenCV (opencv-python)
- PyAV (av)
- PyTorch (with CUDA support if available)
- Ultralytics YOLO (ultralytics)
- Flask
- SciPy
- Chart.js (included via CDN in templates)
- SQLite (using the standard sqlite3 module)

**For example, install using pip:**

```pip install opencv-python av torch ultralytics flask scipy

Configure settings:

**Edit the launch_config.json file or use the "Change launch settings" option in the menu. An example configuration:**

```{
    "rtsp_url": "rtsp://admin:password@10.88.39.16:1051/cam/realmonitor?channel=1&subtype=0",
    "object_type": "car",
    "analysis_mode": "directional",
    "line_options": {
        "orientation": "vertical",
        "position": 0.35,
        "direction_mode": "horizontal"
    },
    "resolution_width": 1600,
    "resolution_height": 900,
    "counting_algorithm": "threshold",
    "min_displacement": 5,
    "record_interval": 60
}

**Run the main application:**

```python main.py

Use the menu to launch the analysis or change settings.

**Run the web analytics interface (optional):**

```python webapp.py

The web interface is available at http://127.0.0.1:5000.

**Example Usage**

Use the Change launch settings menu to select or enter an RTSP URL (either from the database or manually), choose the object type (e.g., car or person), select the analysis mode (directional or unique), choose the counting algorithm (standard or threshold), and set the threshold and record interval.
Launch the analysis with Launch analysis with saved settings.
The analysis runs in real time, updating the database every 60 seconds (by default) with a new session record (or updating the existing session for the same stream and session_start).
View analytics using the View analytics (console) option or via the web interface.

**Debugging**

All key events and errors are logged to debug.log.
If the web interface (diagram page) displays errors (e.g., "Object of type Row is not JSON serializable"), ensure that your database query returns dictionaries (the code uses conn.row_factory = sqlite3.Row and converts rows to dictionaries).
