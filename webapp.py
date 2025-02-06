from flask import Flask, render_template, request, jsonify
import sqlite3
import datetime

app = Flask(__name__)
DB_NAME = "analytics.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

@app.template_filter('datetimeformat')
def datetimeformat(value, format="%d-%m-%Y %H:%M:%S"):
    try:
        dt = datetime.datetime.strptime(value, "%d-%m-%Y %H:%M:%S")
    except Exception:
        dt = datetime.datetime.strptime(value.split(" ")[0], "%Y-%m-%d")
    return dt.strftime(format)

@app.route("/")
def index():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT stream_url, object_type, total, session_start
        FROM analytics
        ORDER BY session_start DESC
    ''')
    streams = cursor.fetchall()
    # Group by stream_url
    streams_dict = {}
    for row in streams:
        stream = row["stream_url"]
        if stream not in streams_dict:
            streams_dict[stream] = {"object_type": row["object_type"],
                                    "session_count": 0,
                                    "first_session": row["session_start"]}
        streams_dict[stream]["session_count"] += 1
    streams_list = []
    for stream, data in streams_dict.items():
        streams_list.append({
            "stream_url": stream,
            "object_type": data["object_type"],
            "session_count": data["session_count"],
            "first_session": data["first_session"]
        })
    conn.close()
    return render_template("index.html", streams=streams_list)

@app.route("/stream/<path:stream_url>")
def stream_details(stream_url):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT object_type, direction1, direction2, total, session_start, session_end
        FROM analytics
        WHERE stream_url = ?
        ORDER BY session_start DESC
    ''', (stream_url,))
    sessions = cursor.fetchall()
    sessions = [dict(row) for row in sessions]
    conn.close()
    return render_template("stream_details.html", stream_url=stream_url, sessions=sessions)

@app.route("/diagram", methods=["GET", "POST"])
def diagram():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT session_start FROM analytics ORDER BY session_start')
    dates = [row["session_start"].split(" ")[0] for row in cursor.fetchall()]
    dates = sorted(list(set(dates)))
    selected_date = request.form.get("selected_date", "")
    sessions = []
    if selected_date:
        cursor.execute('''
            SELECT stream_url, object_type, direction1, direction2, total, session_start, session_end
            FROM analytics
            WHERE session_start LIKE ?
            ORDER BY session_start ASC
        ''', (selected_date + "%",))
        sessions = cursor.fetchall()
        sessions = [dict(row) for row in sessions]
    conn.close()
    return render_template("diagram.html", dates=dates, sessions=sessions, selected_date=selected_date)

@app.route("/data")
def data():
    selected_date = request.args.get("date", "")
    conn = get_db_connection()
    cursor = conn.cursor()
    if selected_date:
        cursor.execute('''
            SELECT stream_url, object_type, direction1, direction2, total, session_start, session_end
            FROM analytics
            WHERE session_start LIKE ?
            ORDER BY session_start ASC
        ''', (selected_date + "%",))
        sessions = cursor.fetchall()
        sessions = [dict(row) for row in sessions]
    else:
        sessions = []
    conn.close()
    return jsonify(sessions)

if __name__ == "__main__":
    app.run(debug=True)
