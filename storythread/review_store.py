"""Review file reads and the one-row decision write.

Used by review_ui. Kept separate so the page module stays the server.
"""
import json
import threading
from pathlib import Path

from .data import rows
from .http_ui import read_lines, write_atomic

STATUSES = ("pending", "skipped", "approved")
SUGGESTION_KEYS = (
    "judge_model", "status", "total", "start", "end", "keep_reason",
    "concern",     "skip_reason", "flags", "name")


def status_of(row):
    """Returns pending, skipped, or approved for a review row.

    Rows written before review_status existed map approved true to approved
    and every other value to pending.
    """
    status = row.get("review_status")
    if status in STATUSES:
        return status
    if row.get("approved") is True:
        return "approved"
    return "pending"


def primary_suggestion(items):
    """Returns the latest suggestion that has a score, else the latest."""
    scored = [item for item in items if type(item.get("total")) is int]
    if scored:
        return scored[-1]
    if items:
        return items[-1]
    return None


def validate_window(turns, start, end):
    """Raises ValueError unless [start, end) ends user, then assistant.

    end is exclusive, matching review.jsonl and data.py prepare.
    """
    count = len(turns)
    if not 0 <= start < end <= count:
        raise ValueError(
            f"Range {start}:{end} is outside 0:{count} (end is exclusive).")
    user_then_assistant = (
        end - start >= 2
        and turns[end - 2].get("role") == "user"
        and turns[end - 1].get("role") == "assistant")
    if not user_then_assistant:
        raise ValueError(
            "Approved window must end with a user turn "
            "followed by an assistant reply.")


def suggestion_view(row):
    """Copies the suggestion fields the page displays."""
    return {key: row[key] for key in SUGGESTION_KEYS if key in row}


def load_scenes(index_path, review_path):
    """Loads name and turns for scene ids that appear in the review file."""
    wanted = set()
    if Path(review_path).is_file():
        for row in rows(review_path):
            if row.get("scene_id"):
                wanted.add(row["scene_id"])
    found = {}
    if not wanted or not Path(index_path).is_file():
        return found
    for row in rows(index_path):
        scene_id = row.get("scene_id")
        if scene_id in wanted and scene_id not in found:
            found[scene_id] = {
                "name": row.get("name") or "",
                "turns": row.get("turns") or [],
            }
            if len(found) == len(wanted):
                break
    return found


class Store:
    """Reads the index and suggestions, and writes one review row at a time."""

    def __init__(self, review, index, suggestions):
        self.review_path = Path(review)
        self.index_path = Path(index)
        self.suggestions_path = Path(suggestions)
        self.lock = threading.Lock()
        self._scenes = None

    def scenes(self):
        """Returns cached index scenes for the current review ids."""
        if self._scenes is None:
            with self.lock:
                if self._scenes is None:
                    self._scenes = load_scenes(
                        self.index_path, self.review_path)
        return self._scenes

    def review_rows(self):
        """Returns every review row, in file order."""
        if not self.review_path.is_file():
            return []
        return list(rows(self.review_path))

    def suggestions(self):
        """Groups suggestion rows by scene_id, in file order."""
        grouped = {}
        if not self.suggestions_path.is_file():
            return grouped
        for row in rows(self.suggestions_path):
            scene_id = row.get("scene_id")
            if scene_id:
                grouped.setdefault(scene_id, []).append(row)
        return grouped

    def turns(self, scene_id):
        """Returns indexed turns, or None when the scene is absent."""
        scene = self.scenes().get(scene_id)
        if scene is None:
            return None
        return scene.get("turns") or []

    def queue(self):
        """Returns the left-pane rows, including a score when one exists."""
        scenes = self.scenes()
        suggestions = self.suggestions()
        items = []
        for order, row in enumerate(self.review_rows()):
            scene_id = row.get("scene_id")
            packed = suggestions.get(scene_id, [])
            primary = primary_suggestion(packed) or {}
            scene = scenes.get(scene_id, {})
            score = primary.get("total")
            start, end = primary.get("start"), primary.get("end")
            items.append({
                "order": order,
                "scene_id": scene_id,
                "name": scene.get("name") or primary.get("name") or "",
                "status": status_of(row),
                "score": score if type(score) is int else None,
                "suggested_start": start if type(start) is int else None,
                "suggested_end": end if type(end) is int else None,
                "judge_model": primary.get("judge_model") or "",
                "suggestion_count": len(packed),
            })
        return items

    def scene_view(self, scene_id):
        """Returns one scene, its window, and any judge suggestions."""
        row = next((item for item in self.review_rows()
                    if item.get("scene_id") == scene_id), None)
        if row is None:
            raise KeyError(scene_id)
        scene = self.scenes().get(scene_id, {})
        turns = scene.get("turns") or []
        packed = self.suggestions().get(scene_id, [])
        system = row.get("system")
        if not isinstance(system, str):
            system = ""
        return {
            "scene_id": scene_id,
            "name": scene.get("name") or "",
            "status": status_of(row),
            "approved": row.get("approved") is True,
            "group": row.get("group") or "",
            "system": system,
            "start": row.get("start"),
            "end": row.get("end"),
            "turns": [{"role": turn.get("role") or "",
                       "content": turn.get("content") or ""}
                      for turn in turns],
            "suggestions": [suggestion_view(item) for item in packed],
        }

    def decide(self, scene_id, system, start, end, status):
        """Saves system, start, end, review_status, and approved.

        approved is true only when status is approved. Other rows are copied
        unchanged. Approve checks the window before anything is written.
        """
        if status not in STATUSES:
            raise ValueError("status must be pending, skipped, or approved.")
        if not isinstance(system, str):
            raise ValueError("system must be a string.")
        if type(start) is not int or type(end) is not int:
            raise ValueError("start and end must be integers.")
        if status == "approved":
            turns = self.turns(scene_id)
            if turns is None:
                raise ValueError("Scene is not in the index.")
            validate_window(turns, start, end)
        with self.lock:
            lines, newline, trailing = read_lines(self.review_path)
            found = False
            updated = []
            for line in lines:
                if not line.strip():
                    updated.append(line)
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        "Review file has a line that is not JSON.") from None
                if row.get("scene_id") != scene_id:
                    updated.append(line)
                    continue
                row["system"] = system
                row["start"] = start
                row["end"] = end
                row["review_status"] = status
                row["approved"] = status == "approved"
                updated.append(json.dumps(row, ensure_ascii=False))
                found = True
            if not found:
                raise KeyError(scene_id)
            text = newline.join(updated)
            if trailing:
                text += newline
            write_atomic(self.review_path, text)
        print(f"{status} {scene_id}", flush=True)
