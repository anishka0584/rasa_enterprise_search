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
# Resolve project root as the folder containing this actions/ directory.
# Using __file__ (not os.getcwd()) makes this robust regardless of what
# directory 'rasa run actions' is invoked from.
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

        # Treat a line as a question/entry if it ends with "?" (standard FAQ format)
        # OR is a short statement-style entry without "?" like "can i order pizza"
        # or "i want to watch a movie". These exist in the FAQ as valid entries.
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

    # Synonym groups — any word in a group matches any other in the same group.
    # Keep groups tight: overly broad synonyms cause wrong matches.
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
        # Normalise hyphens so "wi-fi" → "wifi", "check-in" → "checkin"
        text = re.sub(r'(\w)-(\w)', r'\1\2', text)
        return set(re.findall(r"\w+", text.lower()))

    def expand_tokens(tokens: set) -> set:
        expanded = set(tokens)
        for token in list(tokens):
            for group in SYNONYMS:
                if token in group:
                    expanded |= group
        return expanded

    # Words too generic to be useful for matching — cause wrong FAQ hits
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

        # Bidirectional synonym intersection — only count terms present in at
        # least one side raw (prevents pure-synonym ghost matches)
        intersection = (exp_user & exp_faq) & (raw_user | raw_faq)
        union        = raw_user | raw_faq
        jaccard      = len(intersection) / len(union)   if union     else 0.0
        precision    = len(intersection) / len(raw_user) if raw_user else 0.0

        seq = difflib.SequenceMatcher(
            None, user_question.lower(), pair["q"].lower()
        ).ratio()

        score = 0.35 * jaccard + 0.45 * precision + 0.20 * seq

        if score > best_score:
            best_score  = score
            best_answer = pair["a"]

    # 0.28 threshold — works well with synonym expansion + bidirectional scoring
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
# Shared question handler — used by ALL question-handling actions
# ---------------------------------------------------------------------------

def handle_question(user_message: str, dispatcher: CollectingDispatcher) -> bool:
    """
    Look up user_message in the FAQ. If found, reply with the answer.
    If not found, store it in the DB for the admin and send an acknowledgement.
    Returns True if the message was handled (answered or stored), False if empty.
    Always handles the message regardless of whether a flow is active.
    """
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
    Heuristic to decide if a user message is a genuine question or statement
    rather than a plain slot answer (a date, number, city name, yes/no etc.)
    or a flow-trigger phrase (book a hotel, find me a room, etc.).
    """
    t = text.strip().lower()
    if not t:
        return False

    # Flow-trigger phrases — these start the booking flow, not questions to answer
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

    # Correction/update phrases — slot corrections mid-flow, not FAQ questions.
    # These trigger pattern_correction and must never fire handle_question.
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

    # Cancel/abort phrases — handled by pattern_cancel_flow, not FAQ
    cancel_prefixes = (
        "cancel", "stop the booking", "i want to cancel",
        "i don't want to book", "i do not want to book",
        "forget it", "never mind", "abort",
    )
    for prefix in cancel_prefixes:
        if t.startswith(prefix):
            return False

    # Plain slot answers — not questions
    slot_patterns = [
        r'^\d+$',                          # bare number: "4"
        r'^\d+[\/\-]\d+',                  # date fragment: "05/12"
        r'^\d+(st|nd|rd|th)',              # ordinal: "5th"
        r'^(yes|no|yeah|nope|yep|nah|ok|okay|sure|correct|confirm|confirmed)$',
    ]
    for pat in slot_patterns:
        if re.match(pat, t, re.IGNORECASE):
            return False

    # Compound slot answers — comma-separated values like "4 rooms, 8 people, in goa"
    # or "goa, 5th dec, 8 guests". These are multi-slot answers, never FAQ questions.
    # Detect: message contains commas AND has numbers mixed with words (no question words).
    if "," in t:
        # Strip commas and check if all meaningful tokens are slot-answer material
        no_comma = re.sub(r"[,]", " ", t)
        tokens_nc = no_comma.split()
        question_words = {"what","when","where","who","why","how","which","is","are",
                          "do","does","can","could","will","would","should","have","has"}
        # If none of the tokens are question words, treat as compound slot answer
        if not any(tok in question_words for tok in tokens_nc):
            return False

    # Explicit question markers
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

    # Statement-style FAQ entries without "?" (e.g. "can i order pizza",
    # "i want to watch a movie"). These are 3+ words that aren't slot answers
    # or booking triggers.
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
        handle_question(user_message, dispatcher)
        return []


# ---------------------------------------------------------------------------
# ActionCheckQuestion — injected after every collect step in flows.yml
# Intercepts any question the user asks mid-flow and answers/logs it,
# regardless of which slot is currently being collected.
# ---------------------------------------------------------------------------

class ActionCheckQuestion(Action):

    CANCEL_PHRASES = (
        "cancel booking", "cancel my booking", "i want to cancel booking",
        "i want to cancel my booking", "i want to cancel", "cancel",
        "stop the booking", "stop booking", "abort", "forget it",
        "never mind", "i changed my mind", "i do not want to book",
        "i don't want to book", "drop it", "end this", "quit", "exit",
    )

    def name(self) -> Text:
        return "action_check_question"

    HUMAN_AGENT_PHRASES = (
        "talk to an agent", "speak to an agent", "speak with an agent",
        "talk to a human", "speak to a human", "speak with a human",
        "talk to a person", "speak to a person", "connect me to an agent",
        "i want an agent", "i need an agent", "human agent", "real person",
        "live agent", "talk to support", "speak to support",
        "talk to someone", "speak to someone",
        "i want to talk to", "i want to speak to", "i want to speak with",
    )

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker:    Tracker,
        domain:     Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        user_message = (tracker.latest_message.get("text") or "").strip()
        t = user_message.lower()

        # --- Human agent request ---
        if any(phrase in t for phrase in self.HUMAN_AGENT_PHRASES):
            dispatcher.utter_message(
                text="I understand you'd like to speak with a human agent. "
                     "Unfortunately, live agent support isn't available right now. "
                     "Please contact us directly at support@hotel.com or call +1-800-HOTEL. "
                     "Is there anything else I can help you with?"
            )
            return []

        # --- Cancel interception ---
        if any(t == p or t.startswith(p) for p in self.CANCEL_PHRASES):
            dispatcher.utter_message(
                text="Sorry to see you go! 😊 Your booking has been cancelled. "
                     "Come back anytime — I'm always here to help you find the perfect hotel. "
                     "Have a great day!"
            )
            return [AllSlotsReset(), SlotSet("confirm_booking", "cancelled")]

        # Don't fire FAQ handler if a pattern_correction or pattern_repeat is active.
        try:
            stack = tracker.stack or []
            for frame in stack:
                if isinstance(frame, dict) and frame.get("type") in (
                    "pattern_correction", "pattern_repeat_bot_messages"
                ):
                    return []
        except Exception:
            pass

        if is_question(user_message):
            handle_question(user_message, dispatcher)

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


def strip_llm_garbage(raw: str) -> str:
    """
    Strip common garbage the LLM prepends to slot values, e.g.:
    'to "25th Dec'  →  '25th Dec'
    'to "5th Jan'   →  '5th Jan'
    'set to 5 Jan'  →  '5 Jan'
    """
    s = raw.strip()
    # Remove leading: to ", to ', set to, changed to, update to, etc.
    s = re.sub(r'^(changed?\s+to|updated?\s+to|set\s+to|to)\s*["\']?', '', s, flags=re.IGNORECASE)
    # Remove stray leading quotes/punctuation
    s = re.sub(r'^["\'\s]+', '', s)
    return s.strip()


def try_parse_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    # Strip LLM garbage before parsing
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


# ---------------------------------------------------------------------------
# Slot validators
# ---------------------------------------------------------------------------

# Markers that identify LLM-hallucinated garbage slot values.
# These must be silently nulled and NEVER passed to handle_question/store_unknown_question.
_HALLUCINATION_MARKERS = (
    "undefined", " # ", "wait for", "before setting",
    "explicit confirmation", "do not fill", "never fill",
)

def _intercept_question(raw_slot_value, tracker, dispatcher) -> bool:
    """
    Called at the top of every slot validator.
    If the user's actual message (or the garbled slot value the LLM produced)
    looks like a question, answer/log it via handle_question() and return True
    so the validator knows to null the slot silently without re-asking.
    The flow's own utter_ask_* will fire on the next turn automatically.
    """
    # Guard: if the slot value is LLM-hallucinated garbage (copied from slot description),
    # null it silently — NEVER log it as an unknown question.
    # This catches "undefined\"  # Wait for explicit confirmation before setting" etc.
    if raw_slot_value and isinstance(raw_slot_value, str):
        rv = raw_slot_value.strip().lower()
        if any(marker in rv for marker in _HALLUCINATION_MARKERS):
            return True   # signal: null the slot, stay completely silent

    # The real user message always takes priority over the slot value.
    # IMPORTANT: only call handle_question if the message is a genuine question —
    # never for plain slot answers like dates, numbers, or city names.
    user_message = (tracker.latest_message.get("text") or "").strip()
    if is_question(user_message):
        handle_question(user_message, dispatcher)
        return True

    # Fallback: sometimes the LLM stuffs the user question into the slot value
    # (e.g. num_guests = "is there a taxi?"). Catch that too.
    # But NEVER call handle_question on a value that looks like a slot answer.
    if raw_slot_value and isinstance(raw_slot_value, str):
        val = raw_slot_value.strip()
        if is_question(val) and not parse_number(val):
            handle_question(val, dispatcher)
            return True

    return False


class ValidateNumGuests(Action):

    def name(self) -> Text:
        return "action_validate_num_guests"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("num_guests")

        # CRITICAL FALLBACK: if LLM routed the message to cannot_handle and
        # never set the slot, try to parse the raw user message directly.
        # This catches cases where bare numbers like "4", "24" go to cannot_handle.
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

        # CRITICAL FALLBACK: read raw user message if LLM never set the slot
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
        raw = tracker.get_slot("confirm_booking")
        user_text = (tracker.latest_message.get("text") or "").strip()
        t = user_text.lower()

        # Guard: silently null any LLM-hallucinated garbage value before anything else.
        # Ollama copies slot description text verbatim, e.g.:
        #   "undefined\"  # Wait for explicit confirmation before setting"
        # This must never reach _intercept_question or be logged as an unknown question.
        if raw and isinstance(raw, str):
            rv = raw.strip().lower()
            if any(marker in rv for marker in _HALLUCINATION_MARKERS):
                return [SlotSet("confirm_booking", None)]

        # Human agent request at confirmation step
        if any(phrase in t for phrase in self.HUMAN_AGENT_PHRASES):
            dispatcher.utter_message(
                text="I understand you'd like to speak with a human agent. "
                     "Unfortunately, live agent support isn't available right now. "
                     "Please contact us directly at support@hotel.com or call +1-800-HOTEL. "
                     "Is there anything else I can help you with?"
            )
            return []

        # Cancel detection at confirmation step
        if any(t == p or t.startswith(p) for p in self.CANCEL_DETECT):
            dispatcher.utter_message(
                text="Sorry to see you go! 😊 Your booking has been cancelled. "
                     "Come back anytime — I'm always here to help you find the perfect hotel. "
                     "Have a great day!"
            )
            return [AllSlotsReset(), SlotSet("confirm_booking", "cancelled")]

        if raw is None:
            return []

        # Pass the cancelled sentinel straight through so the flow's next
        # condition can route to step_booking_cancelled without re-asking.
        if str(raw).strip().lower() == "cancelled":
            return [SlotSet("confirm_booking", "cancelled")]

        if _intercept_question(raw, tracker, dispatcher):
            return [SlotSet("confirm_booking", None)]

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

        # If slot value is LLM garbage that won't parse, fall back to the actual
        # user message text for the slot that was just collected this turn.
        # We detect which slot was just filled by checking which one changed.
        user_text = (tracker.latest_message.get("text") or "").strip()

        date_in  = try_parse_date(raw_in)
        date_out = try_parse_date(raw_out)

        # If check_in failed to parse and user_text looks like a date, use it
        if date_in is None and user_text:
            fallback = try_parse_date(user_text)
            if fallback:
                date_in = fallback
                raw_in  = user_text

        # If check_out failed to parse and user_text looks like a date, use it
        if date_out is None and user_text:
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

        # Cross-year fix: if check-out landed before check-in (e.g. check-in Feb 2027,
        # check-out "5 jan" parsed as Jan 2026/2027 before Feb), try the next year.
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
    """Validates num_guests immediately after collect — before flow advances."""

    def name(self) -> Text:
        return "action_validate_guests_now"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("num_guests")

        # Fallback: if validator couldn't save the slot through collect machinery,
        # try the raw user message one more time
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
    """Validates num_rooms immediately after collect — before flow advances."""

    def name(self) -> Text:
        return "action_validate_rooms_now"

    def run(self, dispatcher, tracker, domain):
        raw = tracker.get_slot("num_rooms")

        # Fallback: if validator couldn't save the slot through collect machinery,
        # try the raw user message one more time
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

        # Validate num_guests
        # Use "INVALID" sentinel so pattern_collect_information doesn't fire a
        # separate re-ask on top of our combined error+re-ask utter message.
        # The flow's next condition checks for "INVALID" and loops back to collect.
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

        # Validate num_rooms
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
        # Set confirm_booking to "cancelled" sentinel so that if pattern_collect_information
        # resumes after this interrupting pattern ends, the flow's next condition
        # routes to step_booking_cancelled (silent action_session_end) instead of
        # re-asking the confirmation question. AllSlotsReset alone doesn't stop
        # the collect frame from resuming and re-uttering utter_ask_confirm_booking.
        return [AllSlotsReset(), SlotSet("confirm_booking", "cancelled")]


class ActionHandleCannotHandle(Action):
    """
    Intercepts pattern_cannot_handle when the LLM fails to route a valid
    slot answer (date, number, city name) and instead routes it to cannot_handle.
    
    Checks what slot is currently being collected and tries to fill it directly
    from the raw user message. If the message looks like a valid slot value,
    fills the slot and stays silent (the flow will advance normally).
    If not a valid slot value, falls back to utter_ask_rephrase.
    """

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

    # Phrases that mean the user explicitly wants a human agent
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

        # 1. Human agent request — show before everything else
        if any(phrase in t for phrase in self.HUMAN_AGENT_PHRASES):
            dispatcher.utter_message(
                text="I understand you'd like to speak with a human agent. "
                     "Unfortunately, live agent support isn't available right now. "
                     "Please contact us directly at support@hotel.com or call +1-800-HOTEL. "
                     "Is there anything else I can help you with?"
            )
            return []

        # 2. Cancel commands
        if any(t == p or t.startswith(p) for p in self.CANCEL_PHRASES):
            dispatcher.utter_message(
                text="Sorry to see you go! 😊 Your booking has been cancelled. "
                     "Come back anytime — I'm always here to help you find the perfect hotel. "
                     "Have a great day!"
            )
            return [AllSlotsReset(), SlotSet("confirm_booking", "cancelled")]

        # 3. Try to fill the currently collected slot from the raw user message
        collected_slot = self._get_current_collect_slot(tracker)

        if collected_slot:
            if collected_slot in self.DATE_SLOTS:
                parsed = try_parse_date(user_text)
                if parsed:
                    return [SlotSet(collected_slot, user_text)]

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

        # 4. Answer FAQ / log unknown question — covers human_handoff misrouting
        if is_question(user_text):
            handle_question(user_text, dispatcher)
            return []

        # 5. Genuine cannot-handle — show rephrase
        dispatcher.utter_message(response="utter_ask_rephrase")
        return []

    def _get_current_collect_slot(self, tracker) -> Optional[str]:
        """
        Walk the dialogue stack to find the innermost pattern_collect_information
        frame and return its 'collect' field (the slot currently being asked for).
        """
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
    
class ActionHumanHandoff(Action):
    """
    Called by pattern_human_handoff when user explicitly asks for a human agent.
    Shows a message explaining that live support is unavailable and provides
    contact details. Only triggers for genuine agent requests, not FAQ questions.
    """

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

        # Only show the human handoff message if this really is an agent request.
        # If the LLM incorrectly routed a FAQ question here, answer it instead.
        if any(phrase in user_text for phrase in self.HUMAN_AGENT_PHRASES):
            dispatcher.utter_message(
                text="I understand you'd like to speak with a human agent. "
                     "Unfortunately, live agent support isn't available right now. "
                     "Please contact us directly at support@hotel.com or call +1-800-HOTEL. "
                     "Is there anything else I can help you with?"
            )
            return []

        # LLM misrouted a FAQ question to human handoff — answer it properly
        if is_question(tracker.latest_message.get("text", "")):
            handle_question(tracker.latest_message.get("text", ""), dispatcher)
            return []

        # Fallback for anything else that ends up here
        dispatcher.utter_message(
            text="I understand you'd like to speak with a human agent. "
                 "Unfortunately, live agent support isn't available right now. "
                 "Please contact us at support@hotel.com or call +1-800-HOTEL."
        )
        return []


class ActionFreeChitchat(Action):
    """
    Called by pattern_chitchat. Delegates entirely to handle_question() —
    the same logic used by pattern_search and action_check_question.
    If the message looks like a plain slot answer (double-tagged by the LLM),
    is_question() will return False and we stay silent.
    """

    def name(self) -> Text:
        return "action_free_chitchat"

    def run(self, dispatcher, tracker, domain):
        user_message = (tracker.latest_message.get("text") or "").strip()
        if is_question(user_message):
            handle_question(user_message, dispatcher)
        return []


# ---------------------------------------------------------------------------
# ActionApplyCorrection — called by pattern_correction
# Parses a correction command like "change location to Goa" or
# "update check-in to 10th Jan" and sets the relevant slot directly,
# without relying on the LLM to extract the value.
# ---------------------------------------------------------------------------

class ActionApplyCorrection(Action):
    """
    Handles mid-flow slot corrections.

    Supported phrases (examples):
      change location to London
      update check-in to 10th Jan
      set rooms to 3
      change city to Paris
      update guests to 4
      change check-out to 8th Dec
    """

    def name(self) -> Text:
        return "action_apply_correction"

    # Maps canonical slot names to the regex aliases the user might say
    SLOT_ALIASES: Dict[str, List[str]] = {
        "location":  ["location", "city", "destination", "place", "town"],
        "check_in":  ["check.?in", "checkin", "arrival", "arriving", "start date", "from date", "from"],
        "check_out": ["check.?out", "checkout", "departure", "leaving", "end date", "to date", "until"],
        "num_guests": ["guests?", "people", "persons?", "adults?", "travell?ers?", "number of guests?"],
        "num_rooms":  ["rooms?", "number of rooms?"],
    }

    # Verb prefixes the user might use
    _VERB = r"(?:change|update|set|make|correct|switch|i (?:want to )?change|can you change|please change)"
    _THE  = r"(?:the |my )?"

    def _build_pattern(self) -> re.Pattern:
        """Build a single regex that matches any correction command."""
        alias_groups = []
        for slot, aliases in self.SLOT_ALIASES.items():
            group = "(?:" + "|".join(aliases) + ")"
            alias_groups.append((slot, group))

        # e.g.  change [the] <slot_alias> to <value>
        #        set <slot_alias> to <value>
        parts = []
        for slot, group in alias_groups:
            parts.append(
                rf"(?P<slot_{slot.replace('-','_')}>{group})"
            )

        all_slots = "|".join(p for p in parts)
        pattern_str = (
            rf"^{self._VERB}\s+{self._THE}(?:{all_slots})\s+to\s+(?P<value>.+)$"
        )
        return re.compile(pattern_str, re.IGNORECASE)

    def run(self, dispatcher, tracker, domain):
        user_text = (tracker.latest_message.get("text") or "").strip()
        t = user_text.lower()

        slot_name  = None
        slot_value = None

        try:
            pattern = self._build_pattern()
            m = pattern.match(user_text)
            if m:
                slot_value = (m.group("value") or "").strip()
                # Find which named group matched
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

        # Fallback: simpler keyword scan if regex didn't match
        if not slot_name or not slot_value:
            slot_name, slot_value = self._fallback_parse(t, user_text)

        if not slot_name or not slot_value:
            dispatcher.utter_message(
                text="Sorry, I couldn't understand what you'd like to change. "
                     "Try something like: 'change location to London' or "
                     "'update check-in to 5th Jan'."
            )
            return []

        # Normalise value for date / number slots
        events = self._build_slot_events(slot_name, slot_value, dispatcher)

        if events:
            dispatcher.utter_message(response="utter_corrected_previous_input")

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fallback_parse(self, t: str, original: str):
        """
        Simple keyword scan as a fallback when the main regex fails.
        Returns (slot_name, raw_value) or (None, None).
        """
        # Detect "to <value>" at the end
        to_match = re.search(r'\bto\s+(.+)$', original, re.IGNORECASE)
        if not to_match:
            return None, None
        value = to_match.group(1).strip()

        # Which slot keyword appears before "to"?
        before_to = original[: to_match.start()].lower()

        ordered_checks = [
            ("check_in",  ["check-in", "checkin", "arrival", "check in", "arriving"]),
            ("check_out", ["check-out", "checkout", "departure", "check out", "leaving"]),
            ("num_guests",["guest", "guests", "people", "person", "persons", "traveler", "travellers"]),
            ("num_rooms", ["room", "rooms"]),
            ("location",  ["location", "city", "destination", "place", "town"]),
        ]

        for slot, keywords in ordered_checks:
            if any(kw in before_to for kw in keywords):
                return slot, value

        return None, None

    def _build_slot_events(
        self, slot_name: str, raw_value: str, dispatcher: CollectingDispatcher
    ) -> List[Dict[Text, Any]]:
        """Validate and normalise the value, then return a SlotSet event list."""

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

        else:  # location — text slot
            # Capitalise each word for a clean display value
            clean = raw_value.strip().title()
            if not clean:
                dispatcher.utter_message(
                    text="I couldn't understand the location. Please try again."
                )
                return []
            return [SlotSet(slot_name, clean)]