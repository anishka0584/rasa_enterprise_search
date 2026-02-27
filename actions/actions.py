from typing import Any, Text, Dict, List, Optional
import re
import os
import sqlite3
import difflib
from datetime import datetime
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, AllSlotsReset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
DB_PATH  = os.path.join(_PROJECT_ROOT, "unknown_questions.db")
FAQ_PATH = os.path.join(_PROJECT_ROOT, "docs", "hotel_faq.txt")

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
    pairs = []
    current_q = None
    current_a = []

    for line in faq_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.isupper() and len(stripped.split()) <= 6:
            if current_q and current_a:
                pairs.append({"q": current_q, "a": " ".join(current_a).strip()})
            current_q = None
            current_a = []
            continue

        is_question_line = (
            len(stripped) < 250 and
            not stripped.isupper() and
            (stripped.endswith("?") or len(stripped.split()) <= 10)
        )

        if is_question_line:
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

    SYNONYMS: List[set] = [
        {"wifi", "internet", "wireless"},
        {"taxi", "cab"},
        {"shuttle", "transfer", "transport"},
        {"breakfast", "meal"},
        {"pool", "swimming", "swim"},
        {"gym", "fitness", "workout"},
        {"pet", "pets", "dog", "cat", "animal"},
        {"cancel", "cancellation"},
        {"checkin", "arrive", "arrival"},
        {"checkout", "departure"},
        {"room", "rooms", "suite", "accommodation"},
        {"guest", "guests", "people", "person"},
        {"fee", "charge", "fees"},
        {"children", "child", "kids", "kid", "baby", "infant"},
        {"booking", "reservation", "book", "reserve"},
    ]

    def tokenize_text(text: str) -> set:
        text = re.sub(r'(\w)-(\w)', r'\1\2', text)
        return set(re.findall(r"\w+", text.lower()))

    def expand_tokens(tokens: set) -> set:
        expanded = set(tokens)
        for token in list(tokens):
            for group in SYNONYMS:
                if token in group:
                    expanded |= group
        return expanded

    stop_words = {
        "i", "a", "an", "the", "is", "are", "there", "do", "you",
        "have", "can", "what", "which", "how", "in", "at", "to",
        "for", "of", "my", "me", "it", "be", "will", "does", "your",
        "this", "that", "any", "some", "get", "has", "would", "could",
        "should", "please", "want", "need", "like", "tell", "know",
        "available", "provide", "offer", "allowed", "included",
        "bring", "maximum", "number",
    }

    raw_user = tokenize_text(user_question) - stop_words
    exp_user = expand_tokens(raw_user)

    best_score  = 0.0
    best_answer = None

    for pair in qa_pairs:
        raw_faq = tokenize_text(pair["q"]) - stop_words
        exp_faq = expand_tokens(raw_faq)

        if not raw_user or not raw_faq:
            continue

        intersection = (exp_user & exp_faq) & (raw_user | raw_faq)
        union        = raw_user | raw_faq
        jaccard      = len(intersection) / len(union)    if union    else 0.0
        precision    = len(intersection) / len(raw_user) if raw_user else 0.0

        seq = difflib.SequenceMatcher(
            None, user_question.lower(), pair["q"].lower()
        ).ratio()

        score = 0.35 * jaccard + 0.45 * precision + 0.20 * seq

        if score > best_score:
            best_score  = score
            best_answer = pair["a"]

    return best_answer if best_score >= 0.28 else None


CATEGORY_KEYWORDS = {
    "Pet Policy": ["pet", "dog", "cat", "animal"],
    "Room Types & Amenities": ["room", "wifi", "ac", "bed", "view", "facility"],
    "Check-in & Check-out": ["checkin", "checkout", "check-in", "check-out", "time"],
    "Booking Rules": ["book", "booking", "reservation", "limit"],
    "Cancellations & Changes": ["cancel", "change", "modify"],
    "Payment & Pricing": ["price", "pay", "payment", "fee", "cost"],
    "Children & Family": ["child", "kid", "baby", "family"],
    "General": ["where", "which", "what"],
    "Other": []
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
# Shared question handler
# ---------------------------------------------------------------------------

def handle_question(user_message: str, dispatcher: CollectingDispatcher) -> bool:
    if not user_message:
        return False

    answer = search_faq(user_message)
    if answer:
        dispatcher.utter_message(text=answer)
    else:
        store_unknown_question(user_message)
        dispatcher.utter_message(
            text=(
                "That's a great question, but I don't have a specific answer for it yet. "
                "I've noted it down and our team will look into it. "
                "In the meantime, is there anything else I can help you with?"
            )
        )
    return True


def is_question(text: str) -> bool:
    """
    Returns True if text is a genuine FAQ question/statement,
    False if it is a slot answer, booking trigger, correction, or cancel phrase.
    """
    t = text.strip().lower()
    if not t:
        return False

    booking_triggers = {
        "book a hotel", "book hotel", "book a room", "book room",
        "i want to book", "i want a hotel", "i want a room",
        "find me a hotel", "find a hotel", "find me a room",
        "hotel booking", "make a booking", "make a reservation",
        "i need a hotel", "i need a room", "get me a hotel",
        "start booking", "new booking",
    }
    if any(t == trigger or t.startswith(trigger) for trigger in booking_triggers):
        return False

    correction_prefixes = (
        "change ", "update ", "switch ", "set ", "make it ",
        "correct ", "i meant ", "actually ",
        "i want to change", "i'd like to change", "i would like to change",
        "i want to update", "i'd like to update",
        "i want to switch", "can you change", "can you update",
        "change the ", "update the ", "change my ", "update my ",
        "change location", "change city", "change check", "change date",
        "change number", "change guest", "change room",
    )
    for prefix in correction_prefixes:
        if t.startswith(prefix):
            return False

    cancel_prefixes = (
        "cancel", "stop the booking", "i want to cancel",
        "i don't want to book", "i do not want to book",
        "forget it", "never mind", "abort",
    )
    for prefix in cancel_prefixes:
        if t.startswith(prefix):
            return False

    slot_patterns = [
        r'^\d+$',
        r'^\d+[\/\-]\d+',
        r'^\d+(st|nd|rd|th)',
        r'^(yes|no|yeah|nope|yep|nah|ok|okay|sure|correct|confirm|confirmed)$',
    ]
    for pat in slot_patterns:
        if re.match(pat, t, re.IGNORECASE):
            return False

    if "," in t:
        no_comma = re.sub(r"[,]", " ", t)
        tokens_nc = no_comma.split()
        question_words = {"what","when","where","who","why","how","which","is","are",
                          "do","does","can","could","will","would","should","have","has"}
        if not any(tok in question_words for tok in tokens_nc):
            return False

    question_starters = (
        "what", "when", "where", "who", "why", "how", "which",
        "is ", "are ", "do ", "does ", "can ", "could ", "will ",
        "would ", "should ", "have ", "has ", "is there", "do you",
        "do i", "am i", "i would", "i'd like",
    )
    if t.endswith("?"):
        return True
    for starter in question_starters:
        if t.startswith(starter):
            return True

    words = t.split()
    if len(words) >= 3:
        date_words = {"jan","feb","mar","apr","may","jun","jul","aug",
                      "sep","oct","nov","dec","january","february","march",
                      "april","june","july","august","september","october",
                      "november","december","monday","tuesday","wednesday",
                      "thursday","friday","saturday","sunday","next","tonight",
                      "tomorrow"}
        number_words = {"one","two","three","four","five","six","seven",
                        "eight","nine","ten","eleven","twelve"}
        meaningful = [w for w in words if w not in ("i","a","the","to","of","me")]
        if all(w in date_words | number_words for w in meaningful):
            return False
        return True

    return False


# ---------------------------------------------------------------------------
# Helpers: number parsing, date parsing
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

_MONTH_RE = re.compile(
    r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b',
    re.IGNORECASE
)

# Keywords that identify a message as a booking trigger (used when scanning events)
_BOOKING_TRIGGER_KEYWORDS = (
    "book a hotel", "book hotel", "book a room", "book room",
    "i want to book", "i want a hotel", "find me a hotel", "find a hotel",
    "hotel booking", "make a booking", "make a reservation",
    "i need a hotel", "i need a room", "get me a hotel",
)


def normalise_date_string(raw: str) -> str:
    s = ORDINAL_RE.sub(r'\1', raw)
    s = re.sub(r'[,]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def strip_llm_garbage(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r'^(changed?\s+to|updated?\s+to|set\s+to|to)\s*["\']?', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^["\'\s]+', '', s)
    return s.strip()


def try_parse_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    raw = strip_llm_garbage(str(raw))
    cleaned = normalise_date_string(raw)
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


def get_booking_trigger_message(tracker: Tracker) -> str:
    """
    Find the user message that triggered the book_hotel flow.

    THE CORE FIX for the 'hi → book a hotel from X to Y' bug:

    When the user says 'hi' first, the bot greets them. Then the user says
    'book a hotel from 4th to 6th jan for 5 people'. By the time
    action_check_question runs (after collect: location asks "Which city?"),
    tracker.latest_message may already be the CITY answer, not the booking
    trigger. The trigger is one or more events back in tracker.events.

    This function walks tracker.events newest-first and returns the most recent
    user message that looks like a booking trigger. Falls back to
    latest_message if none is found.
    """
    latest = (tracker.latest_message.get("text") or "").strip()

    # Fast path: current message IS the trigger
    if any(kw in latest.lower() for kw in _BOOKING_TRIGGER_KEYWORDS):
        return latest

    # Walk events newest-first to find the trigger
    try:
        for event in reversed(tracker.events):
            if event.get("event") != "user":
                continue
            text = (event.get("text") or "").strip()
            if any(kw in text.lower() for kw in _BOOKING_TRIGGER_KEYWORDS):
                return text
    except Exception:
        pass

    return latest


def extract_slots_from_text(text: str) -> Dict[str, Optional[str]]:
    """
    Parse a natural-language booking message and extract any slot values it
    contains using pure regex — no LLM required. Returns a dict with keys:
    check_in, check_out, num_guests, num_rooms, location. Values are
    formatted strings or None.

    Handles phrases like:
      'book a hotel from 4th to 6th jan for 5 people'
      'i want a room in goa from 4th jan to 9th jan for 3 people 2 rooms'
      'book from 1st to 7th march for 4 guests 2 rooms in paris'
      'hotel in london 5th dec to 10th dec'
    """
    result: Dict[str, Optional[str]] = {
        "check_in":  None,
        "check_out": None,
        "num_guests": None,
        "num_rooms":  None,
        "location":   None,
    }

    t = text.strip()

    # --- Dates: "from/between X to/and Y" ---
    date_patterns = [
        r'(?:from|between)\s+(\d+(?:st|nd|rd|th)?(?:\s+[a-zA-Z]+)?)\s+(?:to|and|-)\s+(\d+(?:st|nd|rd|th)?(?:\s+[a-zA-Z]+)?)',
        r'(\d+(?:st|nd|rd|th)?)\s+(?:to|-)\s+(\d+(?:st|nd|rd|th)?(?:\s+[a-zA-Z]+)?)',
    ]
    found_date_range = False
    for pat in date_patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            raw_in  = m.group(1).strip()
            raw_out = m.group(2).strip()

            # Borrow month between the two sides if one is missing it
            if not _MONTH_RE.search(raw_in):
                mm = _MONTH_RE.search(raw_out)
                if mm:
                    raw_in = raw_in + " " + mm.group()

            if not _MONTH_RE.search(raw_out):
                mm = _MONTH_RE.search(raw_in)
                if mm:
                    raw_out = raw_out + " " + mm.group()

            d_in  = try_parse_date(raw_in)
            d_out = try_parse_date(raw_out)

            if d_in and d_out and d_out > d_in:
                result["check_in"]  = d_in.strftime("%d/%m/%Y")
                result["check_out"] = d_out.strftime("%d/%m/%Y")
                found_date_range = True
            break

    # --- Single check-in date: "from 4th jan" / "starting 5th dec" / "arriving 10th jan" ---
    # Only attempt this if we didn't already find a full date range above.
    if not found_date_range and result["check_in"] is None:
        single_in_m = re.search(
            r'(?:from|starting|arriving|arriving on|checkin on|check.?in on|check in on)\s+'
            r'(\d+(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|'
            r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?'
            r'|\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?)',
            t, re.IGNORECASE
        )
        if single_in_m:
            d_in = try_parse_date(single_in_m.group(1).strip())
            if d_in:
                result["check_in"] = d_in.strftime("%d/%m/%Y")

    # --- num_rooms: "X room(s)" — check before guests to avoid misparse ---
    rooms_m = re.search(
        r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+rooms?\b',
        t, re.IGNORECASE
    )
    if rooms_m:
        n = parse_number(rooms_m.group(1))
        if n and 1 <= n <= 10:
            result["num_rooms"] = str(n)

    # --- num_guests: "for X people/guests" or "X people/guests" ---
    guests_m = re.search(
        r'\bfor\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|'
        r'nineteen|twenty(?:\s+\w+)?|just me|myself)\s*(?:people|guests?|persons?|adults?|pax)?\b',
        t, re.IGNORECASE
    )
    if not guests_m:
        guests_m = re.search(
            r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)'
            r'\s+(?:people|guests?|persons?|adults?|pax)\b',
            t, re.IGNORECASE
        )
    if guests_m:
        n = parse_number(guests_m.group(1))
        if n and 1 <= n <= 24:
            result["num_guests"] = str(n)

    # --- location: "in <city>" ---
    location_m = re.search(
        r'\bin\s+([A-Za-z][a-zA-Z\s]{1,29}?)(?:\s+from|\s+for|\s+\d|\s+on\b|,|$)',
        t, re.IGNORECASE
    )
    if location_m:
        loc = location_m.group(1).strip()
        # Reject if it's a month name alone or starts with a digit
        if loc and not re.match(r'^\d', loc) and not _MONTH_RE.fullmatch(loc):
            result["location"] = loc.title()

    return result


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

_HALLUCINATION_MARKERS = (
    "undefined",
    " # ",
    "#",
    "wait for",
    "before setting",
    "explicit confirmation",
    "do not fill",
    "never fill",
    "not provided yet",
    'null"',
    "null'",
    "not yet",
    "placeholder",
    "to be filled",
    "leave blank",
    "leave null",
    "tbd",
)


def is_hallucinated(value) -> bool:
    """Return True if a slot value is LLM-hallucinated garbage."""
    if not value or not isinstance(value, str):
        return False
    rv = value.strip().lower()
    if not rv:
        return False
    if any(marker in rv for marker in _HALLUCINATION_MARKERS):
        return True
    if rv.startswith("null") and len(rv) > 4:
        return True
    if re.match(r'^["\']?(null|undefined)["\']?\s*[,;#\s]', rv):
        return True
    return False


# ---------------------------------------------------------------------------
# _intercept_question helper
# ---------------------------------------------------------------------------

def _intercept_question(raw_slot_value, tracker, dispatcher) -> bool:
    """
    Called at the top of every slot validator.
    Returns True if the slot should be nulled (hallucination or genuine question).
    """
    if is_hallucinated(raw_slot_value):
        return True

    user_message = (tracker.latest_message.get("text") or "").strip()
    if is_question(user_message):
        handle_question(user_message, dispatcher)
        return True

    if raw_slot_value and isinstance(raw_slot_value, str):
        val = raw_slot_value.strip()
        if is_question(val) and not parse_number(val):
            handle_question(val, dispatcher)
            return True

    return False


# ---------------------------------------------------------------------------
# ActionTriggerSearch — called by pattern_search
# ---------------------------------------------------------------------------

class ActionTriggerSearch(Action):

    def name(self) -> Text:
        return "action_trigger_search"

    def run(self, dispatcher, tracker, domain):
        user_message = (tracker.latest_message.get("text") or "").strip()
        handle_question(user_message, dispatcher)
        return []


# ---------------------------------------------------------------------------
# ActionCheckQuestion
#
# Runs after every collect step in flows.yml. Responsibilities:
#   1. Human-agent and cancel interception
#   2. Sanitise hallucinated slot values from initial LLM extraction
#   3. Pre-fill missing slots from the booking trigger message via regex.
#      Uses get_booking_trigger_message() which scans tracker.events to find
#      the trigger even when prior turns (hi, chitchat) preceded it.
#      This is the fix for: hi → greet → "book hotel from 4th to 6th jan for 5 people"
#      not pre-filling dates/guests.
#   4. Answer FAQ questions asked mid-flow
# ---------------------------------------------------------------------------

class ActionCheckQuestion(Action):

    CANCEL_PHRASES = (
        "cancel booking", "cancel my booking", "i want to cancel booking",
        "i want to cancel my booking", "i want to cancel", "cancel",
        "stop the booking", "stop booking", "abort", "forget it",
        "never mind", "i changed my mind", "i do not want to book",
        "i don't want to book", "drop it", "end this", "quit", "exit",
    )

    HUMAN_AGENT_PHRASES = (
        "talk to an agent", "speak to an agent", "speak with an agent",
        "talk to a human", "speak to a human", "speak with a human",
        "talk to a person", "speak to a person", "connect me to an agent",
        "i want an agent", "i need an agent", "human agent", "real person",
        "live agent", "talk to support", "speak to support",
        "talk to someone", "speak to someone",
        "i want to talk to", "i want to speak to", "i want to speak with",
    )

    def name(self) -> Text:
        return "action_check_question"

    def run(self, dispatcher, tracker, domain):

        user_message = (tracker.latest_message.get("text") or "").strip()
        t = user_message.lower()

        # 1. Human agent request
        if any(phrase in t for phrase in self.HUMAN_AGENT_PHRASES):
            dispatcher.utter_message(
                text="I understand you'd like to speak with a human agent. "
                     "Unfortunately, live agent support isn't available right now. "
                     "Please contact us directly at support@hotel.com or call +1-800-HOTEL. "
                     "Is there anything else I can help you with?"
            )
            return []

        # 2. Cancel interception
        if any(t == p or t.startswith(p) for p in self.CANCEL_PHRASES):
            dispatcher.utter_message(
                text="Sorry to see you go! 😊 Your booking has been cancelled. "
                     "Come back anytime — I'm always here to help you find the perfect hotel. "
                     "Have a great day!"
            )
            return [AllSlotsReset(), SlotSet("confirm_booking", "cancelled")]

        # 3. Sanitise hallucinated garbage slot values.
        #    IMPORTANT: only emit SlotSet(slot, None) when the slot currently holds
        #    a non-null hallucinated value. If it is already null, emitting
        #    SlotSet(slot, None) still creates a slot_was_set event in the tracker,
        #    which causes Rasa's pattern_collect_information to treat that slot as
        #    "touched this turn" and skip the utter_ask_* question entirely.
        #
        #    CRITICAL: Never clean up `location` here. action_check_question runs
        #    immediately after `collect: location`, so if we emit SlotSet("location", None)
        #    the tracker sees a slot_was_set event for location this turn and Rasa's
        #    pattern_collect_information skips asking "Which city?" entirely.
        #    Location hallucination is handled by ValidateLocation instead, which runs
        #    inside the collect step where the skip behaviour is intentional.
        cleanup_events = []
        for slot_name in ("check_in", "check_out", "num_guests", "num_rooms"):
            val = tracker.get_slot(slot_name)
            if val is not None and is_hallucinated(val):
                cleanup_events.append(SlotSet(slot_name, None))

        # Determine which slots are null after cleanup (candidates for pre-filling).
        # A slot is a candidate if it was null already OR we just nulled it above.
        cleaned_names = {e["name"] for e in cleanup_events}
        null_slots = set()
        for slot_name in ("location", "check_in", "check_out", "num_guests", "num_rooms"):
            val = tracker.get_slot(slot_name)
            if val is None or slot_name in cleaned_names:
                null_slots.add(slot_name)

        # 4. Pre-fill missing slots from the booking trigger message using regex.
        #
        #    get_booking_trigger_message() scans tracker.events backwards to find
        #    the message that triggered book_hotel, even if it was not the most
        #    recent turn (e.g. after a prior "hi" greeting exchange).
        #
        #    CRITICAL: Never pre-fill `location` here.
        #    `action_check_question` runs AFTER `collect: location` — meaning the
        #    flow already entered the location collect step, decided location was null,
        #    and is waiting for it. If we emit SlotSet("location", <value>) now, the
        #    collect machinery sees a slot event for location in this turn and skips
        #    asking the question. Since the user didn't provide a city, we must leave
        #    location null so the bot correctly asks "Which city?".
        #    Location can only be pre-filled if the user actually said it in their
        #    message (e.g. "book a hotel in goa from...") — in that case the LLM
        #    extractor will have already set it before we get here, so it won't be
        #    in null_slots anyway.
        PRE_FILL_SLOTS = {"check_in", "check_out", "num_guests", "num_rooms"}

        if null_slots & PRE_FILL_SLOTS:
            trigger_text = get_booking_trigger_message(tracker)
            extracted    = extract_slots_from_text(trigger_text)
            for slot_name, value in extracted.items():
                if value and slot_name in null_slots and slot_name in PRE_FILL_SLOTS:
                    cleanup_events.append(SlotSet(slot_name, value))

        # 5. FAQ handler — skip when inside a correction/repeat pattern
        try:
            for frame in (tracker.stack or []):
                if isinstance(frame, dict) and frame.get("type") in (
                    "pattern_correction", "pattern_repeat_bot_messages"
                ):
                    return cleanup_events
        except Exception:
            pass

        if is_question(user_message):
            handle_question(user_message, dispatcher)

        return cleanup_events


# ---------------------------------------------------------------------------
# Slot validators
# ---------------------------------------------------------------------------

class ValidateNumGuests(Action):

    def name(self) -> Text:
        return "action_validate_num_guests"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("num_guests")

        user_text = (tracker.latest_message.get("text") or "").strip()
        if raw is None and user_text:
            n = parse_number(user_text)
            if n is not None:
                raw = str(n)

        if raw is None:
            return []

        if _intercept_question(raw, tracker, dispatcher):
            return [SlotSet("num_guests", None)]

        guests = parse_number(raw)
        if guests is None:
            return [SlotSet("num_guests", None)]

        if guests > 24:
            dispatcher.utter_message(response="utter_too_many_guests")
            return [SlotSet("num_guests", None)]

        return [SlotSet("num_guests", str(guests))]


class ValidateNumRooms(Action):

    def name(self) -> Text:
        return "action_validate_num_rooms"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("num_rooms")

        user_text = (tracker.latest_message.get("text") or "").strip()
        if raw is None and user_text:
            n = parse_number(user_text)
            if n is not None:
                raw = str(n)

        if raw is None:
            return []

        if _intercept_question(raw, tracker, dispatcher):
            return [SlotSet("num_rooms", None)]

        rooms = parse_number(raw)
        if rooms is None:
            return [SlotSet("num_rooms", None)]

        if rooms > 10:
            dispatcher.utter_message(response="utter_too_many_rooms")
            return [SlotSet("num_rooms", None)]

        return [SlotSet("num_rooms", str(rooms))]


class ValidateLocation(Action):
    """
    Runs inside the collect: location step (wired via flows.yml).
    Clears hallucinated location values so Rasa re-asks the question.

    WHY THIS EXISTS:
    action_check_question cannot safely emit SlotSet("location", None) because
    it runs *after* the collect step exits — any slot_was_set event for location
    at that point tricks Rasa's pattern_collect_information into skipping
    utter_ask_location. This validator runs *inside* the collect step, so the
    skip behaviour is correct: if we null it here Rasa will ask again.
    """

    def name(self) -> Text:
        return "validate_location"

    def run(self, dispatcher, tracker, domain):
        val = tracker.get_slot("location")
        if val is None:
            return []
        if is_hallucinated(val):
            return [SlotSet("location", None)]
        # Basic sanity: reject if it looks like a date or a bare number
        v = val.strip()
        if re.match(r'^\d', v) or try_parse_date(v):
            return [SlotSet("location", None)]
        return []


class ValidateConfirmBooking(Action):

    def name(self) -> Text:
        return "validate_confirm_booking"

    HUMAN_AGENT_PHRASES = (
        "talk to an agent", "speak to an agent", "speak with an agent",
        "talk to a human", "speak to a human", "speak with a human",
        "talk to a person", "connect me to an agent", "i want an agent",
        "i need an agent", "human agent", "real person", "live agent",
        "i want to talk to", "i want to speak to", "i want to speak with",
    )

    CANCEL_DETECT = (
        "cancel booking", "cancel my booking", "i want to cancel booking",
        "i want to cancel my booking", "i want to cancel", "cancel",
        "stop the booking", "stop booking", "abort", "forget it",
        "never mind", "i changed my mind", "i do not want to book",
        "i don't want to book", "drop it", "end this", "quit", "exit",
    )

    def run(self, dispatcher, tracker, domain):
        raw       = tracker.get_slot("confirm_booking")
        user_text = (tracker.latest_message.get("text") or "").strip()
        t         = user_text.lower()

        if is_hallucinated(raw):
            return [SlotSet("confirm_booking", None)]

        if any(phrase in t for phrase in self.HUMAN_AGENT_PHRASES):
            dispatcher.utter_message(
                text="I understand you'd like to speak with a human agent. "
                     "Unfortunately, live agent support isn't available right now. "
                     "Please contact us directly at support@hotel.com or call +1-800-HOTEL. "
                     "Is there anything else I can help you with?"
            )
            return []

        if any(t == p or t.startswith(p) for p in self.CANCEL_DETECT):
            dispatcher.utter_message(
                text="Sorry to see you go! 😊 Your booking has been cancelled. "
                     "Come back anytime — I'm always here to help you find the perfect hotel. "
                     "Have a great day!"
            )
            return [AllSlotsReset(), SlotSet("confirm_booking", "cancelled")]

        if raw is None:
            return []

        if str(raw).strip().lower() == "cancelled":
            return [SlotSet("confirm_booking", "cancelled")]

        if _intercept_question(raw, tracker, dispatcher):
            return [SlotSet("confirm_booking", None)]

        yes_values = {"true", "yes", "yeah", "yep", "correct", "confirm",
                      "confirmed", "ok", "okay", "sure", "looks good",
                      "that's right", "thats right", "right"}
        no_values  = {"false", "no", "nope", "nah", "wrong", "incorrect",
                      "change", "update", "edit", "not right"}

        # Strip LLM prefix garbage like "to true", "set to true", "changed to yes"
        normalised = re.sub(
            r'^(set\s+to|changed?\s+to|updated?\s+to|to)\s+', '', str(raw).strip().lower()
        ).strip()

        if normalised in yes_values:
            return [SlotSet("confirm_booking", "true")]
        if normalised in no_values:
            return [SlotSet("confirm_booking", "false")]

        # Fallback: check the user's actual message directly.
        # This catches cases where the LLM garbles the slot value entirely
        # but the user plainly said "yes" or "no".
        user_lower = t.strip()
        if user_lower in yes_values or user_lower.startswith("yes") or user_lower.startswith("yeah"):
            return [SlotSet("confirm_booking", "true")]
        if user_lower in no_values or user_lower.startswith("no"):
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

        if is_hallucinated(raw_in):
            return [SlotSet("check_in", None)]
        if is_hallucinated(raw_out):
            return [SlotSet("check_out", None)]

        date_in  = try_parse_date(raw_in)
        date_out = try_parse_date(raw_out)

        # Apply user_text fallback only to the ONE slot that failed to parse.
        # Never apply to both — that causes check_in = check_out = user_text
        # which triggers an infinite "must be after" validation loop.
        user_text = (tracker.latest_message.get("text") or "").strip()

        if date_in is None and date_out is not None and user_text:
            fallback = try_parse_date(user_text)
            if fallback:
                date_in = fallback
                raw_in  = user_text

        elif date_out is None and date_in is not None and user_text:
            fallback = try_parse_date(user_text)
            if fallback:
                date_out = fallback
                raw_out  = user_text

        events = []

        if date_in is None:
            dispatcher.utter_message(
                text=f"I couldn't understand the check-in date '{strip_llm_garbage(raw_in)}'. "
                     "Please enter a date like '5 Dec 2026' or '05/12/2026'."
            )
            events.append(SlotSet("check_in", None))

        if date_out is None:
            dispatcher.utter_message(
                text=f"I couldn't understand the check-out date '{strip_llm_garbage(raw_out)}'. "
                     "Please enter a date like '8 Dec 2026' or '08/12/2026'."
            )
            events.append(SlotSet("check_out", None))

        if events:
            return events

        # Cross-year fix
        if date_in >= date_out:
            advanced = date_out.replace(year=date_out.year + 1)
            if advanced > date_in:
                date_out = advanced
            else:
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


class ActionValidateGuestsNow(Action):

    def name(self) -> Text:
        return "action_validate_guests_now"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("num_guests")

        user_text = (tracker.latest_message.get("text") or "").strip()
        if raw is None and user_text and not is_question(user_text):
            n = parse_number(user_text)
            if n is not None:
                if n > 24:
                    dispatcher.utter_message(response="utter_too_many_guests")
                    return [SlotSet("num_guests", None)]
                return [SlotSet("num_guests", str(n))]

        if raw is None:
            return []
        n = parse_number(raw)
        if n is not None and n > 24:
            dispatcher.utter_message(response="utter_too_many_guests")
            return [SlotSet("num_guests", None)]
        return []


class ActionValidateRoomsNow(Action):

    def name(self) -> Text:
        return "action_validate_rooms_now"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("num_rooms")

        user_text = (tracker.latest_message.get("text") or "").strip()
        if raw is None and user_text and not is_question(user_text):
            n = parse_number(user_text)
            if n is not None:
                if n > 10:
                    dispatcher.utter_message(response="utter_too_many_rooms")
                    return [SlotSet("num_rooms", None)]
                return [SlotSet("num_rooms", str(n))]

        if raw is None:
            return []
        n = parse_number(raw)
        if n is not None and n > 10:
            dispatcher.utter_message(response="utter_too_many_rooms")
            return [SlotSet("num_rooms", None)]
        return []


class ActionFormatNumbers(Action):

    def name(self) -> Text:
        return "action_format_numbers"

    def run(self, dispatcher, tracker, domain):
        events = []

        guests_raw = tracker.get_slot("num_guests")
        if guests_raw is not None and guests_raw != "INVALID":
            guests = parse_number(guests_raw)
            if guests is None:
                events.append(SlotSet("num_guests", "INVALID"))
            elif guests > 24:
                dispatcher.utter_message(response="utter_too_many_guests")
                events.append(SlotSet("num_guests", "INVALID"))
            else:
                events.append(SlotSet("num_guests", str(guests)))

        rooms_raw = tracker.get_slot("num_rooms")
        if rooms_raw is not None and rooms_raw != "INVALID":
            rooms = parse_number(rooms_raw)
            if rooms is None:
                events.append(SlotSet("num_rooms", "INVALID"))
            elif rooms > 10:
                dispatcher.utter_message(response="utter_too_many_rooms")
                events.append(SlotSet("num_rooms", "INVALID"))
            else:
                events.append(SlotSet("num_rooms", str(rooms)))

        return events


# ---------------------------------------------------------------------------
# ActionSessionEnd
# ---------------------------------------------------------------------------

class ActionSessionEnd(Action):

    def name(self) -> Text:
        return "action_session_end"

    def run(self, dispatcher, tracker, domain):
        return []


# ---------------------------------------------------------------------------
# ActionCancelBooking
# ---------------------------------------------------------------------------

class ActionCancelBooking(Action):

    def name(self) -> Text:
        return "action_cancel_booking"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(
            text="Sorry to see you go! 😊 Your booking has been cancelled. "
                 "Come back anytime — I'm always here to help you find the perfect hotel. "
                 "Have a great day!"
        )
        return [AllSlotsReset(), SlotSet("confirm_booking", "cancelled")]


# ---------------------------------------------------------------------------
# ActionHandleCannotHandle — called by pattern_cannot_handle
# ---------------------------------------------------------------------------

class ActionHandleCannotHandle(Action):

    def name(self) -> Text:
        return "action_handle_cannot_handle"

    DATE_SLOTS   = {"check_in", "check_out"}
    NUMBER_SLOTS = {"num_guests", "num_rooms"}
    TEXT_SLOTS   = {"location"}

    CANCEL_PHRASES = {
        "cancel", "stop", "quit", "exit", "abort", "forget it",
        "never mind", "i changed my mind", "i do not want to book",
        "i don't want to book", "cancel booking", "cancel my booking",
        "i want to cancel", "i want to cancel booking",
        "i want to cancel my booking", "stop the booking",
        "drop it", "end this",
    }

    HUMAN_AGENT_PHRASES = (
        "talk to an agent", "speak to an agent", "speak with an agent",
        "talk to a human", "speak to a human", "speak with a human",
        "talk to a person", "speak to a person", "connect me to an agent",
        "connect me with an agent", "i want an agent", "i need an agent",
        "human agent", "real person", "live agent", "talk to support",
        "speak to support", "connect me to support", "talk to someone",
        "speak to someone", "i want to talk to",
        "i want to speak to", "i want to speak with",
    )

    def run(self, dispatcher, tracker, domain):
        user_text = (tracker.latest_message.get("text") or "").strip()
        t = user_text.lower()

        if any(phrase in t for phrase in self.HUMAN_AGENT_PHRASES):
            dispatcher.utter_message(
                text="I understand you'd like to speak with a human agent. "
                     "Unfortunately, live agent support isn't available right now. "
                     "Please contact us directly at support@hotel.com or call +1-800-HOTEL. "
                     "Is there anything else I can help you with?"
            )
            return []

        if any(t == p or t.startswith(p) for p in self.CANCEL_PHRASES):
            dispatcher.utter_message(
                text="Sorry to see you go! 😊 Your booking has been cancelled. "
                     "Come back anytime — I'm always here to help you find the perfect hotel. "
                     "Have a great day!"
            )
            return [AllSlotsReset(), SlotSet("confirm_booking", "cancelled")]

        collected_slot = self._get_current_collect_slot(tracker)

        if collected_slot:
            if collected_slot in self.DATE_SLOTS:
                parsed = try_parse_date(user_text)
                if parsed:
                    return [SlotSet(collected_slot, parsed.strftime("%d/%m/%Y"))]

            elif collected_slot in self.NUMBER_SLOTS:
                n = parse_number(user_text)
                if n is not None:
                    if collected_slot == "num_guests" and n > 24:
                        dispatcher.utter_message(response="utter_too_many_guests")
                        return [SlotSet(collected_slot, None)]
                    if collected_slot == "num_rooms" and n > 10:
                        dispatcher.utter_message(response="utter_too_many_rooms")
                        return [SlotSet(collected_slot, None)]
                    return [SlotSet(collected_slot, str(n))]

            elif collected_slot in self.TEXT_SLOTS:
                if user_text and not is_question(user_text) and len(user_text.split()) <= 5:
                    return [SlotSet(collected_slot, user_text.title())]

        if is_question(user_text):
            handle_question(user_text, dispatcher)
            return []

        dispatcher.utter_message(response="utter_ask_rephrase")
        return []

    def _get_current_collect_slot(self, tracker) -> Optional[str]:
        try:
            stack = tracker.stack
            if not stack:
                return None
            for frame in reversed(stack):
                if isinstance(frame, dict) and frame.get("type") == "pattern_collect_information":
                    return frame.get("collect")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# ActionHumanHandoff — called by pattern_human_handoff
# ---------------------------------------------------------------------------

class ActionHumanHandoff(Action):

    HUMAN_AGENT_PHRASES = (
        "talk to an agent", "speak to an agent", "speak with an agent",
        "talk to a human", "speak to a human", "speak with a human",
        "talk to a person", "speak to a person", "connect me to an agent",
        "i want an agent", "i need an agent", "human agent", "real person",
        "live agent", "talk to support", "speak to support",
        "talk to someone", "speak to someone",
        "i want to talk to", "i want to speak to", "i want to speak with",
    )

    def name(self) -> Text:
        return "action_human_handoff"

    def run(self, dispatcher, tracker, domain):
        user_text = (tracker.latest_message.get("text") or "").strip().lower()

        if any(phrase in user_text for phrase in self.HUMAN_AGENT_PHRASES):
            dispatcher.utter_message(
                text="I understand you'd like to speak with a human agent. "
                     "Unfortunately, live agent support isn't available right now. "
                     "Please contact us directly at support@hotel.com or call +1-800-HOTEL. "
                     "Is there anything else I can help you with?"
            )
            return []

        if is_question(tracker.latest_message.get("text", "")):
            handle_question(tracker.latest_message.get("text", ""), dispatcher)
            return []

        dispatcher.utter_message(
            text="I understand you'd like to speak with a human agent. "
                 "Unfortunately, live agent support isn't available right now. "
                 "Please contact us at support@hotel.com or call +1-800-HOTEL."
        )
        return []


# ---------------------------------------------------------------------------
# ActionFreeChitchat — called by pattern_chitchat
# ---------------------------------------------------------------------------

class ActionFreeChitchat(Action):

    def name(self) -> Text:
        return "action_free_chitchat"

    def run(self, dispatcher, tracker, domain):
        user_message = (tracker.latest_message.get("text") or "").strip()
        if is_question(user_message):
            handle_question(user_message, dispatcher)
        return []


# ---------------------------------------------------------------------------
# ActionApplyCorrection — called by pattern_correction
# ---------------------------------------------------------------------------

class ActionApplyCorrection(Action):

    def name(self) -> Text:
        return "action_apply_correction"

    SLOT_ALIASES: Dict[str, List[str]] = {
        "location":   ["location", "city", "destination", "place", "town"],
        "check_in":   ["check.?in", "checkin", "arrival", "arriving", "start date", "from date", "from"],
        "check_out":  ["check.?out", "checkout", "departure", "leaving", "end date", "to date", "until"],
        "num_guests": ["guests?", "people", "persons?", "adults?", "travell?ers?", "number of guests?"],
        "num_rooms":  ["rooms?", "number of rooms?"],
    }

    _VERB = r"(?:change|update|set|make|correct|switch|i (?:want to )?change|can you change|please change)"
    _THE  = r"(?:the |my )?"

    def _build_pattern(self) -> re.Pattern:
        parts = []
        for slot, aliases in self.SLOT_ALIASES.items():
            group = "(?:" + "|".join(aliases) + ")"
            parts.append(rf"(?P<slot_{slot.replace('-','_')}>{group})")
        all_slots   = "|".join(parts)
        pattern_str = rf"^{self._VERB}\s+{self._THE}(?:{all_slots})\s+to\s+(?P<value>.+)$"
        return re.compile(pattern_str, re.IGNORECASE)

    def run(self, dispatcher, tracker, domain):
        user_text  = (tracker.latest_message.get("text") or "").strip()
        t          = user_text.lower()
        slot_name  = None
        slot_value = None

        try:
            m = self._build_pattern().match(user_text)
            if m:
                slot_value = (m.group("value") or "").strip()
                for slot in self.SLOT_ALIASES:
                    gname = f"slot_{slot.replace('-', '_')}"
                    try:
                        if m.group(gname):
                            slot_name = slot
                            break
                    except IndexError:
                        continue
        except Exception as e:
            print(f"[ActionApplyCorrection] regex error: {e}")

        if not slot_name or not slot_value:
            slot_name, slot_value = self._fallback_parse(t, user_text)

        if not slot_name or not slot_value:
            dispatcher.utter_message(
                text="Sorry, I couldn't understand what you'd like to change. "
                     "Try something like: 'change location to London' or "
                     "'update check-in to 5th Jan'."
            )
            return []

        events = self._build_slot_events(slot_name, slot_value, dispatcher)
        if events:
            dispatcher.utter_message(response="utter_corrected_previous_input")
        return events

    def _fallback_parse(self, t: str, original: str):
        to_match = re.search(r'\bto\s+(.+)$', original, re.IGNORECASE)
        if not to_match:
            return None, None
        value     = to_match.group(1).strip()
        before_to = original[: to_match.start()].lower()

        for slot, keywords in [
            ("check_in",   ["check-in", "checkin", "arrival", "check in", "arriving"]),
            ("check_out",  ["check-out", "checkout", "departure", "check out", "leaving"]),
            ("num_guests", ["guest", "guests", "people", "person", "persons", "traveler", "travellers"]),
            ("num_rooms",  ["room", "rooms"]),
            ("location",   ["location", "city", "destination", "place", "town"]),
        ]:
            if any(kw in before_to for kw in keywords):
                return slot, value

        return None, None

    def _build_slot_events(self, slot_name: str, raw_value: str, dispatcher: CollectingDispatcher):
        if slot_name in ("check_in", "check_out"):
            parsed = try_parse_date(raw_value)
            if parsed is None:
                dispatcher.utter_message(
                    text=f"I couldn't understand the date '{raw_value}'. "
                         "Please use a format like '5 Dec 2026' or '05/12/2026'."
                )
                return []
            return [SlotSet(slot_name, parsed.strftime("%d/%m/%Y"))]

        elif slot_name in ("num_guests", "num_rooms"):
            n = parse_number(raw_value)
            if n is None:
                dispatcher.utter_message(
                    text=f"I couldn't understand the number '{raw_value}'. Please give a plain number."
                )
                return []
            if slot_name == "num_guests" and n > 24:
                dispatcher.utter_message(response="utter_too_many_guests")
                return []
            if slot_name == "num_rooms" and n > 10:
                dispatcher.utter_message(response="utter_too_many_rooms")
                return []
            return [SlotSet(slot_name, str(n))]

        else:
            clean = raw_value.strip().title()
            if not clean:
                dispatcher.utter_message(text="I couldn't understand the location. Please try again.")
                return []
            return [SlotSet(slot_name, clean)]