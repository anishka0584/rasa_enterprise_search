from typing import Any, Text, Dict, List, Optional
import re
from datetime import datetime, timedelta
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_number(value: Any) -> Optional[int]:
    """
    Converts any value the LLM might return into a clean integer.
    Handles word numbers, hedged language ('I think 4'), and plain digits.
    Returns None if unparseable.
    """
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

    # Strip hedging phrases so '4 rooms i think' → '4'
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


# Date formats we'll try to parse in order
DATE_FORMATS = [
    "%d/%m/%Y",   # 05/12/2026
    "%d-%m-%Y",   # 05-12-2026
    "%Y-%m-%d",   # 2026-12-05
    "%d %B %Y",   # 05 December 2026
    "%d %b %Y",   # 05 Dec 2026
    "%B %d %Y",   # December 05 2026
    "%b %d %Y",   # Dec 05 2026
    "%d %B",      # 05 December  (no year – assume next occurrence)
    "%d %b",      # 05 Dec
    "%B %d",      # December 05
    "%b %d",      # Dec 05
    "%d/%m",      # 05/12
]

ORDINAL_RE = re.compile(r'(\d+)(st|nd|rd|th)', re.IGNORECASE)


def normalise_date_string(raw: str) -> str:
    """Strip ordinal suffixes and tidy separators so strptime has a better chance."""
    s = ORDINAL_RE.sub(r'\1', raw)          # '5th' → '5'
    s = re.sub(r'[,]', ' ', s)              # remove commas
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def try_parse_date(raw: str) -> Optional[datetime]:
    """Attempt to parse a free-text date string into a datetime. Returns None on failure."""
    if not raw:
        return None

    cleaned = normalise_date_string(str(raw))
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            # If no year was in the format, assume the next upcoming occurrence
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

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

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

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

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
    """
    Normalises the confirm_booking slot to the string 'true' or 'false'.
    """

    def name(self) -> Text:
        return "validate_confirm_booking"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

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


# ---------------------------------------------------------------------------
# Date validation action
# ---------------------------------------------------------------------------

class ValidateDates(Action):
    """
    Validates that:
      1. check_in is parseable
      2. check_out is parseable
      3. check_in is strictly before check_out
      4. The stay is no longer than 14 nights

    If validation fails the offending slot(s) are cleared so the flow re-asks.
    """

    MAX_NIGHTS = 14

    def name(self) -> Text:
        return "validate_dates"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        raw_in  = tracker.get_slot("check_in")
        raw_out = tracker.get_slot("check_out")

        # Nothing to validate yet
        if raw_in is None or raw_out is None:
            return []

        date_in  = try_parse_date(raw_in)
        date_out = try_parse_date(raw_out)

        events = []

        # Unparseable check-in
        if date_in is None:
            dispatcher.utter_message(
                text=f"I couldn't understand the check-in date '{raw_in}'. "
                     "Please enter a date like '5 Dec 2026' or '05/12/2026'."
            )
            events.append(SlotSet("check_in", None))

        # Unparseable check-out
        if date_out is None:
            dispatcher.utter_message(
                text=f"I couldn't understand the check-out date '{raw_out}'. "
                     "Please enter a date like '8 Dec 2026' or '08/12/2026'."
            )
            events.append(SlotSet("check_out", None))

        # If either failed to parse, return early
        if events:
            return events

        # check_in must be strictly before check_out
        if date_in >= date_out:
            dispatcher.utter_message(
                text="Your check-out date must be after your check-in date. "
                     "Please enter a valid check-out date."
            )
            events.append(SlotSet("check_out", None))
            return events

        # Maximum stay length
        nights = (date_out - date_in).days
        if nights > self.MAX_NIGHTS:
            dispatcher.utter_message(
                text=f"Our maximum booking length is {self.MAX_NIGHTS} nights, "
                     f"but your selected stay is {nights} nights. "
                     "Please choose an earlier check-out date."
            )
            events.append(SlotSet("check_out", None))
            return events

        # All good — normalise both dates to a consistent DD/MM/YYYY display format
        events.append(SlotSet("check_in",  date_in.strftime("%d/%m/%Y")))
        events.append(SlotSet("check_out", date_out.strftime("%d/%m/%Y")))
        return events


# ---------------------------------------------------------------------------
# Format action
# ---------------------------------------------------------------------------

class ActionFormatNumbers(Action):
    """Final safety pass before the summary to ensure clean integer display."""

    def name(self) -> Text:
        return "action_format_numbers"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        events = []
        for slot in ["num_guests", "num_rooms"]:
            val = tracker.get_slot(slot)
            if val is not None:
                try:
                    events.append(SlotSet(slot, str(int(float(val)))))
                except (ValueError, TypeError):
                    pass
        return events


# ---------------------------------------------------------------------------
# Session end
# ---------------------------------------------------------------------------

class ActionSessionEnd(Action):
    """Overrides Rasa Pro's built-in pattern_end_of_conversation to stay silent."""

    def name(self) -> Text:
        return "action_session_end"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        return []


class ActionCancelBooking(Action):
    """
    Handles booking cancellation cleanly.
    Sends the cancellation message and resets all booking slots
    so the flow cannot resume after cancellation.
    """

    def name(self) -> Text:
        return "action_cancel_booking"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(
            text="Sorry to see you go! 😊 Your booking has been cancelled. "
                 "Come back anytime — I'm always here to help you find the perfect hotel. "
                 "Have a great day!"
        )

        # Clear all booking slots so the flow cannot resume
        return [
            SlotSet("location", None),
            SlotSet("check_in", None),
            SlotSet("check_out", None),
            SlotSet("num_guests", None),
            SlotSet("num_rooms", None),
            SlotSet("confirm_booking", None),
        ]