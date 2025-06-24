import sqlite3
import os

DB_FILE = "history.db"

def get_conn():
    conn = sqlite3.connect(DB_FILE)
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  result_image TEXT,
                  boxes TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def add_history(filename, result_image, boxes):
    import json
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO history (filename, result_image, boxes) VALUES (?, ?, ?)",
              (filename, result_image, json.dumps(boxes)))
    conn.commit()
    conn.close()

def get_all_history():
    import json
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, filename, result_image, boxes, created_at FROM history ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "filename": row[1],
            "result_image": row[2],
            "boxes": json.loads(row[3]),
            "created_at": row[4]
        })
    return history

init_db()