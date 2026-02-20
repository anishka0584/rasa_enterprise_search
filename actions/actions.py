from typing import Any, Text, Dict, List, Optional
import re
import os
import sqlite3
import difflib
from datetime import datetime
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "..", "unknown_questions.db")
FAQ_PATH = os.path.join(BASE_DIR, "..", "docs", "hotel_faq.txt")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
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

# ---------------------------------------------------------------------------
# FAQ loader & parser
# ---------------------------------------------------------------------------

def load_faq_text() -> str:
    if not os.path.exists(FAQ_PATH):
        return ""
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        return f.read()


def parse_faq_pairs(faq_text: str) -> List[Dict[str, str]]:
    """Parse hotel_faq.txt into a list of {q, a} dicts."""
    pairs = []
    current_q = None
    current_a = []

    for line in faq_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Section headers are short ALL-CAPS lines — skip them
        if stripped.isupper() and len(stripped.split()) <= 6:
            if current_q and current_a:
                pairs.append({"q": current_q, "a": " ".join(current_a).strip()})
            current_q = None
            current_a = []
            continue

        if stripped.endswith("?") and len(stripped) < 250:
            if current_q and current_a:
                pairs.append({"q": current_q, "a": " ".join(current_a).strip()})
            current_q = stripped
            current_a = []
        elif current_q is not None:
            current_a.append(stripped)

    if current_q and current_a:
        pairs.append({"q": current_q, "a": " ".join(current_a).strip()})

    return pairs


def search_faq(user_question: str) -> Optional[str]:
    qa_pairs = parse_faq_pairs(load_faq_text())

    try:
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT question, answer FROM unknown_questions "
            "WHERE status = 'answered' AND answer IS NOT NULL"
        ).fetchall()
        conn.close()
        for row in rows:
            qa_pairs.append({"q": row["question"], "a": row["answer"]})
    except Exception:
        pass

    if not qa_pairs:
        return None

    stop_words = {"i", "a", "an", "the", "is", "are", "there", "do", "you",
                  "have", "can", "what", "which", "how", "in", "at", "to",
                  "for", "of", "my", "me", "it", "be", "will", "does", "your",
                  "this", "that", "any", "some", "get", "has"}

    user_tokens = set(re.findall(r"\w+", user_question.lower())) - stop_words

    best_score  = 0.0
    best_answer = None

    for pair in qa_pairs:
        faq_tokens = set(re.findall(r"\w+", pair["q"].lower())) - stop_words
        if not faq_tokens or not user_tokens:
            continue

        intersection = user_tokens & faq_tokens
        union        = user_tokens | faq_tokens
        jaccard      = len(intersection) / len(union) if union else 0.0

        # Precision: how many of the USER's words appear in the FAQ question
        precision    = len(intersection) / len(user_tokens) if user_tokens else 0.0

        seq = difflib.SequenceMatcher(
            None, user_question.lower(), pair["q"].lower()
        ).ratio()

        # Weighted: precision matters most to avoid wrong matches
        score = 0.5 * jaccard + 0.3 * precision + 0.2 * seq

        if score > best_score:
            best_score  = score
            best_answer = pair["a"]

    # Raised threshold — requires genuine keyword overlap
    return best_answer if best_score >= 0.42 else None

# ---------------------------------------------------------------------------
# Unknown question storage
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Pet Policy":              ["pet", "dog", "cat", "animal", "fur"],
    "Room Types & Amenities":  ["room", "suite", "wifi", "wi-fi", "breakfast",
                                "pool", "gym", "ac", "air", "parking", "tv", "bed"],
    "Check-in & Check-out":    ["check-in", "check in", "check-out", "check out",
                                "arrive", "departure", "early", "late", "time"],
    "Booking Rules":           ["book", "booking", "reservation", "maximum",
                                "limit", "how many", "guest"],
    "Cancellations & Changes": ["cancel", "cancellation", "change", "modify",
                                "refund", "amend"],
    "Payment & Pricing":       ["pay", "payment", "price", "cost", "fee",
                                "charge", "discount", "card"],
    "Children & Family":       ["child", "children", "kid", "family", "baby",
                                "cot", "crib", "infant"],
    "General":                 ["location", "city", "where", "contact",
                                "support", "confirm", "special"],
}


def classify_question(question: str) -> str:
    q = question.lower()
    best, best_count = "Other", 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in q)
        if count > best_count:
            best_count = count
            best = cat
    return best


def store_unknown_question(question: str) -> None:
    try:
        conn = get_db_connection()
        existing = conn.execute(
            "SELECT id FROM unknown_questions WHERE LOWER(question) = LOWER(?)",
            (question,)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE unknown_questions SET asked_count = asked_count + 1 WHERE id = ?",
                (existing["id"],)
            )
        else:
            conn.execute(
                """INSERT INTO unknown_questions
                   (question, category, status, asked_count, created_at)
                   VALUES (?, ?, 'pending', 1, ?)""",
                (question, classify_question(question), datetime.now().isoformat())
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[store_unknown_question] DB error: {e}")


# ---------------------------------------------------------------------------
# ActionTriggerSearch — called by pattern_search
# ---------------------------------------------------------------------------

class ActionTriggerSearch(Action):

    def name(self) -> Text:
        return "action_trigger_search"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker:    Tracker,
        domain:     Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        user_message = (tracker.latest_message.get("text") or "").strip()

        print(f"[DEBUG] FAQ_PATH = {FAQ_PATH}")
        print(f"[DEBUG] FAQ exists = {os.path.exists(FAQ_PATH)}")
        print(f"[DEBUG] FAQ pairs loaded = {len(parse_faq_pairs(load_faq_text()))}")
        print(f"[DEBUG] user_message = {user_message}")
        print(f"[DEBUG] search result = {search_faq(user_message)}")
        
        if not user_message:
            dispatcher.utter_message(
                text="Could you rephrase your question? I want to make sure I help you correctly."
            )
            return []

        answer = search_faq(user_message)

        if answer:
            dispatcher.utter_message(text=answer)
        else:
            store_unknown_question(user_message)
            dispatcher.utter_message(
                text=(
                    "That's a great question, but I don't have a specific answer for it yet. "
                    "I've noted it down and our team will look into it. "
                    "In the meantime, please contact our support team directly via the details "
                    "in your booking confirmation. Is there anything else I can help you with?"
                )
            )

        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_number(value: Any) -> Optional[int]:
    if value is None:
        return None

    word_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'just me': 1, 'myself': 1,
        'alone': 1, 'solo': 1,
    }

    str_val = str(value).strip().lower()
    if str_val in word_map:
        return word_map[str_val]

    str_val = re.sub(
        r'\b(i think|i guess|maybe|approximately|about|around|roughly|perhaps|probably)\b',
        '', str_val
    ).strip()

    match = re.search(r'\d+', str_val)
    if match:
        return int(match.group())

    try:
        return int(float(str_val))
    except (ValueError, TypeError):
        return None


DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
    "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
    "%d %B", "%d %b", "%B %d", "%b %d", "%d/%m",
]

ORDINAL_RE = re.compile(r'(\d+)(st|nd|rd|th)', re.IGNORECASE)


def normalise_date_string(raw: str) -> str:
    s = ORDINAL_RE.sub(r'\1', raw)
    s = re.sub(r'[,]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def try_parse_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    cleaned = normalise_date_string(str(raw))
    today   = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            if "%Y" not in fmt and "%y" not in fmt:
                parsed = parsed.replace(year=today.year)
                if parsed < today:
                    parsed = parsed.replace(year=today.year + 1)
            return parsed
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Slot validators
# ---------------------------------------------------------------------------

class ValidateNumGuests(Action):

    def name(self) -> Text:
        return "validate_num_guests"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("num_guests")
        if raw is None:
            return []

        guests = parse_number(raw)
        if guests is None:
            dispatcher.utter_message(
                text="I didn't catch the number of guests. Please enter a number, for example: 2"
            )
            return [SlotSet("num_guests", None)]

        if guests > 24:
            dispatcher.utter_message(
                text="We can only accommodate up to 24 guests. Please enter a smaller number."
            )
            return [SlotSet("num_guests", None)]

        return [SlotSet("num_guests", str(guests))]


class ValidateNumRooms(Action):

    def name(self) -> Text:
        return "validate_num_rooms"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("num_rooms")
        if raw is None:
            return []

        rooms = parse_number(raw)
        if rooms is None:
            dispatcher.utter_message(
                text="I didn't catch the number of rooms. Please enter a number, for example: 1"
            )
            return [SlotSet("num_rooms", None)]

        if rooms > 10:
            dispatcher.utter_message(
                text="We have a maximum of 10 rooms per booking. Please enter a number between 1 and 10."
            )
            return [SlotSet("num_rooms", None)]

        return [SlotSet("num_rooms", str(rooms))]


class ValidateConfirmBooking(Action):

    def name(self) -> Text:
        return "validate_confirm_booking"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("confirm_booking")
        if raw is None:
            return []

        yes_values = {"true", "yes", "yeah", "yep", "correct", "confirm",
                      "confirmed", "ok", "okay", "sure", "looks good",
                      "that's right", "thats right", "right"}
        no_values  = {"false", "no", "nope", "nah", "wrong", "incorrect",
                      "change", "update", "edit", "not right"}

        normalised = str(raw).strip().lower()

        if normalised in yes_values:
            return [SlotSet("confirm_booking", "true")]
        if normalised in no_values:
            return [SlotSet("confirm_booking", "false")]

        return [SlotSet("confirm_booking", None)]


class ValidateDates(Action):

    MAX_NIGHTS = 14

    def name(self) -> Text:
        return "validate_dates"

    def run(self, dispatcher, tracker, domain):
        raw_in  = tracker.get_slot("check_in")
        raw_out = tracker.get_slot("check_out")

        if raw_in is None or raw_out is None:
            return []

        date_in  = try_parse_date(raw_in)
        date_out = try_parse_date(raw_out)
        events   = []

        if date_in is None:
            dispatcher.utter_message(
                text=f"I couldn't understand the check-in date '{raw_in}'. "
                     "Please enter a date like '5 Dec 2026' or '05/12/2026'."
            )
            events.append(SlotSet("check_in", None))

        if date_out is None:
            dispatcher.utter_message(
                text=f"I couldn't understand the check-out date '{raw_out}'. "
                     "Please enter a date like '8 Dec 2026' or '08/12/2026'."
            )
            events.append(SlotSet("check_out", None))

        if events:
            return events

        if date_in >= date_out:
            dispatcher.utter_message(
                text="Your check-out date must be after your check-in date. "
                     "Please enter a valid check-out date."
            )
            return [SlotSet("check_out", None)]

        nights = (date_out - date_in).days
        if nights > self.MAX_NIGHTS:
            dispatcher.utter_message(
                text=f"Our maximum booking length is {self.MAX_NIGHTS} nights, "
                     f"but your selected stay is {nights} nights. "
                     "Please choose an earlier check-out date."
            )
            return [SlotSet("check_out", None)]

        return [
            SlotSet("check_in",  date_in.strftime("%d/%m/%Y")),
            SlotSet("check_out", date_out.strftime("%d/%m/%Y")),
        ]


class ActionFormatNumbers(Action):

    def name(self) -> Text:
        return "action_format_numbers"

    def run(self, dispatcher, tracker, domain):
        events = []
        for slot in ["num_guests", "num_rooms"]:
            val = tracker.get_slot(slot)
            if val is not None:
                try:
                    events.append(SlotSet(slot, str(int(float(val)))))
                except (ValueError, TypeError):
                    pass
        return events


class ActionSessionEnd(Action):

    def name(self) -> Text:
        return "action_session_end"

    def run(self, dispatcher, tracker, domain):
        return []


class ActionCancelBooking(Action):

    def name(self) -> Text:
        return "action_cancel_booking"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(
            text="Sorry to see you go! 😊 Your booking has been cancelled. "
                 "Come back anytime — I'm always here to help you find the perfect hotel. "
                 "Have a great day!"
        )
        return [
            SlotSet("location", None),
            SlotSet("check_in", None),
            SlotSet("check_out", None),
            SlotSet("num_guests", None),
            SlotSet("num_rooms", None),
            SlotSet("confirm_booking", None),
        ]
    
class ActionFreeChitchat(Action):

    def name(self) -> Text:
        return "action_free_chitchat"

    def run(self, dispatcher, tracker, domain):
        user_message = (tracker.latest_message.get("text") or "").strip()

        # Skip pure slot answers: single word without ?, pure numbers, or dates
        # But allow through anything that looks like a question
        slot_like = (
            re.match(r'^\d+$', user_message) or            # pure number: "4", "96"
            re.match(r'^\d+[\/\-]\d+', user_message) or   # date: "05/12"
            (len(user_message.split()) == 1 and            # single word like "london", "goa"
             not user_message.endswith('?'))
        )
        if slot_like:
            return []

        # Try FAQ first — maybe it's a hotel question the LLM mislabelled as chitchat
        answer = search_faq(user_message)
        if answer:
            dispatcher.utter_message(text=answer)
        else:
            store_unknown_question(user_message)
            dispatcher.utter_message(
                text=(
                    "I'm not sure I can help with that specific question, "
                    "but I've noted it down for our team. "
                    "Is there anything else about your booking I can help with?"
                )
            )
        return []