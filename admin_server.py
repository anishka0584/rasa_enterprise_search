"""
admin_server.py
===============
A lightweight Flask admin panel for managing unknown questions logged
by the Rasa hotel booking bot.

Endpoints
---------
GET  /api/questions          – list all questions (filter by ?status=pending|answered)
GET  /api/questions/<id>     – get a single question
PUT  /api/questions/<id>     – update category / answer / status
DELETE /api/questions/<id>   – delete a question
GET  /api/categories         – list available categories
GET  /api/stats              – summary counts
POST /api/questions/<id>/answer – submit an answer; also writes it into hotel_faq.txt

The HTML UI is served at /  (admin_ui.html embedded inline).

Run with:
    python admin_server.py

Requires:  pip install flask flask-cors
"""

import os
import re
import sqlite3
import json
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, "unknown_questions.db")
# hotel_faq.txt is inside the docs/ folder that Rasa's EnterpriseSearchPolicy indexes
FAQ_PATH  = os.path.join(BASE_DIR, "docs", "hotel_faq.txt")
# Fallback if docs/ doesn't exist yet
if not os.path.exists(os.path.dirname(FAQ_PATH)):
    os.makedirs(os.path.dirname(FAQ_PATH), exist_ok=True)
    # Copy from root if present
    root_faq = os.path.join(BASE_DIR, "hotel_faq.txt")
    if os.path.exists(root_faq) and not os.path.exists(FAQ_PATH):
        import shutil
        shutil.copy(root_faq, FAQ_PATH)

CATEGORIES = [
    "Pet Policy",
    "Room Types & Amenities",
    "Check-in & Check-out",
    "Booking Rules",
    "Cancellations & Changes",
    "Payment & Pricing",
    "Children & Family",
    "General",
    "Other",
]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS unknown_questions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT    NOT NULL,
            category    TEXT    DEFAULT 'Uncategorised',
            answer      TEXT    DEFAULT NULL,
            status      TEXT    DEFAULT 'pending',
            asked_count INTEGER DEFAULT 1,
            created_at  TEXT    NOT NULL,
            answered_at TEXT    DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def row_to_dict(row) -> dict:
    return dict(row)

# ---------------------------------------------------------------------------
# FAQ writer
# ---------------------------------------------------------------------------

def append_to_faq(category: str, question: str, answer: str) -> None:
    """
    Append a learned Q&A pair to hotel_faq.txt under the correct category header.
    If the category section already exists the Q&A is inserted under it.
    If not, a new section is created at the bottom.
    """
    # Read existing content
    if os.path.exists(FAQ_PATH):
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "HOTEL BOOKING ASSISTANT - FAQ KNOWLEDGE BASE\n\n"

    qa_block = f"\n{question}\n{answer}\n"

    # Try to find the category header (case-insensitive)
    header_upper = category.upper()
    pattern = re.compile(
        rf"(^{re.escape(header_upper)}\s*$)",
        re.MULTILINE | re.IGNORECASE
    )

    match = pattern.search(content)
    if match:
        # Insert Q&A right after the header line
        insert_pos = match.end()
        content = content[:insert_pos] + "\n" + qa_block + content[insert_pos:]
    else:
        # Create a new section at the end
        content = content.rstrip() + f"\n\n\n{header_upper}\n{qa_block}"

    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Routes – API
# ---------------------------------------------------------------------------

@app.route("/api/questions", methods=["GET"])
def list_questions():
    status_filter = request.args.get("status")  # optional
    conn = get_conn()
    if status_filter:
        rows = conn.execute(
            "SELECT * FROM unknown_questions WHERE status = ? ORDER BY asked_count DESC, created_at DESC",
            (status_filter,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM unknown_questions ORDER BY asked_count DESC, created_at DESC"
        ).fetchall()
    conn.close()
    return jsonify([row_to_dict(r) for r in rows])


@app.route("/api/questions/<int:qid>", methods=["GET"])
def get_question(qid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM unknown_questions WHERE id = ?", (qid,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Not found"}), 404
    return jsonify(row_to_dict(row))


@app.route("/api/questions/<int:qid>", methods=["PUT"])
def update_question(qid):
    data = request.get_json(force=True)
    allowed = {"category", "answer", "status"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return jsonify({"error": "Nothing to update"}), 400

    conn = get_conn()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values     = list(updates.values()) + [qid]
    conn.execute(f"UPDATE unknown_questions SET {set_clause} WHERE id = ?", values)
    conn.commit()
    row = conn.execute("SELECT * FROM unknown_questions WHERE id = ?", (qid,)).fetchone()
    conn.close()
    return jsonify(row_to_dict(row))


@app.route("/api/questions/<int:qid>", methods=["DELETE"])
def delete_question(qid):
    conn = get_conn()
    conn.execute("DELETE FROM unknown_questions WHERE id = ?", (qid,))
    conn.commit()
    conn.close()
    return jsonify({"deleted": qid})


@app.route("/api/questions/<int:qid>/answer", methods=["POST"])
def answer_question(qid):
    """
    Submit a human-written answer.
    Marks the question as answered, saves the answer in the DB,
    and appends it to hotel_faq.txt so the bot learns it immediately.
    """
    data     = request.get_json(force=True)
    answer   = (data.get("answer") or "").strip()
    category = (data.get("category") or "General").strip()

    if not answer:
        return jsonify({"error": "Answer cannot be empty"}), 400

    now = datetime.now().isoformat()
    conn = get_conn()
    row = conn.execute("SELECT * FROM unknown_questions WHERE id = ?", (qid,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Not found"}), 404

    conn.execute(
        """UPDATE unknown_questions
           SET answer = ?, category = ?, status = 'answered', answered_at = ?
           WHERE id = ?""",
        (answer, category, now, qid)
    )
    conn.commit()
    updated = conn.execute("SELECT * FROM unknown_questions WHERE id = ?", (qid,)).fetchone()
    conn.close()

    # Write into FAQ file so the bot learns it
    append_to_faq(category, row["question"], answer)

    return jsonify(row_to_dict(updated))


@app.route("/api/categories", methods=["GET"])
def get_categories():
    return jsonify(CATEGORIES)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    conn = get_conn()
    total    = conn.execute("SELECT COUNT(*) FROM unknown_questions").fetchone()[0]
    pending  = conn.execute("SELECT COUNT(*) FROM unknown_questions WHERE status = 'pending'").fetchone()[0]
    answered = conn.execute("SELECT COUNT(*) FROM unknown_questions WHERE status = 'answered'").fetchone()[0]
    top = conn.execute(
        "SELECT question, asked_count FROM unknown_questions ORDER BY asked_count DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return jsonify({
        "total":    total,
        "pending":  pending,
        "answered": answered,
        "top_questions": [row_to_dict(r) for r in top],
    })


# ---------------------------------------------------------------------------
# Route – serve the admin UI
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def admin_ui():
    return send_from_directory(BASE_DIR, "admin_ui.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Hotel Bot Admin Panel")
    print("  http://localhost:6060")
    print("=" * 60)
    app.run(host="0.0.0.0", port=6060, debug=True)