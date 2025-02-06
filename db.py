import sqlite3

DB_NAME = "analytics.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Table init with key (stream_url, session_start)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            stream_url TEXT,
            object_type TEXT,
            direction1 INTEGER,
            direction2 INTEGER,
            total INTEGER,
            session_start TEXT,
            session_end TEXT,
            PRIMARY KEY (stream_url, session_start)
        )
    ''')
    conn.commit()
    conn.close()

def upsert_session(rtsp_url, object_type, direction1, direction2, total, session_start, session_end):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analytics (stream_url, object_type, direction1, direction2, total, session_start, session_end)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stream_url, session_start) DO UPDATE SET
                object_type = excluded.object_type,
                direction1 = excluded.direction1,
                direction2 = excluded.direction2,
                total = excluded.total,
                session_end = excluded.session_end
        ''', (rtsp_url, object_type, direction1, direction2, total, session_start, session_end))
        conn.commit()
    except Exception as e:
        print("DB upsert error:", e)
    finally:
        conn.close()

def get_all_sessions():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analytics ORDER BY session_start DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
