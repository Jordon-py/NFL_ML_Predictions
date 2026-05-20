from collections import OrderedDict
import time
from typing import Any, Optional


class LRUCache:
    """A tiny in-process LRU cache with TTL support.

    - max_items: maximum number of items to retain
    - ttl: seconds to keep entries; non-positive disables TTL
    """

    def __init__(self, max_items: int = 256, ttl: int = 300) -> None:
        self.max_items = max(1, int(max_items or 1))
        self.ttl = int(ttl or 0)
        self._store = OrderedDict()  # key -> (ts, value)

    def _expire_if_needed(self, key: str) -> None:
        if key not in self._store:
            return
        ts, _ = self._store.get(key, (0, None))
        if self.ttl > 0 and (time.time() - float(ts)) > float(self.ttl):
            try:
                del self._store[key]
            except Exception:
                pass

    def get(self, key: str) -> Optional[Any]:
        self._expire_if_needed(key)
        val = self._store.get(key)
        if val is None:
            return None
        ts, payload = val
        # Move to end as most recently used
        try:
            self._store.move_to_end(key)
        except Exception:
            pass
        return payload

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            try:
                del self._store[key]
            except Exception:
                pass
        # Evict oldest if over capacity
        while len(self._store) >= self.max_items:
            try:
                self._store.popitem(last=False)
            except Exception:
                break
        try:
            self._store[key] = (time.time(), value)
        except Exception:
            pass

    def pop(self, key: str) -> None:
        try:
            if key in self._store:
                del self._store[key]
        except Exception:
            pass

    def clear(self) -> None:
        try:
            self._store.clear()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._store)

    def items(self):
        return list(self._store.items())
