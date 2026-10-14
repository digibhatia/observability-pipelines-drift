"""Conversation memory, extended from Smriti with explicit thread and user
correlation - not just a session_id, but a way to group several sessions
under the same real customer (user_id), and to mark a narrower thread within
one long session (thread_id), e.g. 'the mortgage sub-topic' within a longer
chat that also covered savings rates."""
from dataclasses import dataclass, field

MAX_TURNS_DEFAULT = 8


@dataclass
class Session:
    session_id: str
    user_id: str = "anonymous"
    thread_id: str = "default"
    turns: list = field(default_factory=list)
    truncated_count: int = 0

    def append(self, role: str, content: str, thread_id: str = None):
        if thread_id:
            self.thread_id = thread_id
        self.turns.append({"role": role, "content": content, "thread_id": self.thread_id})

    def truncate(self, max_turns: int = MAX_TURNS_DEFAULT):
        if len(self.turns) > max_turns:
            overflow = len(self.turns) - max_turns
            self.turns = self.turns[-max_turns:]
            self.truncated_count += overflow


class MemoryStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._crosstalk_shared_session: Session = None  # incident 4 only

    def get_or_create(self, session_id: str, user_id: str = "anonymous") -> Session:
        import incidents
        if incidents.session_crosstalk_active():
            # Incident 4: EVERY session_id is handed back the same underlying
            # Session object, regardless of which id was requested. The
            # .session_id field is cosmetically overwritten so a naive log
            # line still looks right - the turns themselves are shared.
            if self._crosstalk_shared_session is None:
                self._crosstalk_shared_session = Session(session_id=session_id, user_id=user_id)
            self._crosstalk_shared_session.session_id = session_id
            return self._crosstalk_shared_session

        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id, user_id=user_id)
        return self._sessions[session_id]

    def inspect(self, session_id: str) -> dict:
        s = self._sessions.get(session_id)
        if s is None:
            return {"found": False}
        return {
            "found": True,
            "session_id": s.session_id,
            "user_id": s.user_id,
            "thread_id": s.thread_id,
            "turn_count": len(s.turns),
            "truncated_count_lifetime": s.truncated_count,
            "turns": s.turns,
        }

    def reset(self, session_id: str):
        self._sessions.pop(session_id, None)

    def sessions_for_user(self, user_id: str) -> list:
        """All session_ids belonging to the same real customer - the point of
        a separate user_id from session_id: a customer may start a NEW
        session_id each time they open the chat widget, but still be the
        same person across visits."""
        return [sid for sid, s in self._sessions.items() if s.user_id == user_id]


store = MemoryStore()
