"""Per-frame answer key from PFF tags: DEAD / CONTROL / TRAVELLING / LOOSE / UNKNOWN (M1 step 1).

Pure function of plain inputs (no I/O, no project imports): event rows as :class:`Row`, frame arrays, the frame span of
each game event (from the tracking ``game_event`` start_frame/end_frame, else from aligned startTime/endTime) and the
aligned frame of each row's eventTime. Labels use the exact PFF times; nothing is shifted. Comparisons between labels
and tracking use :data:`TOLERANCE_S` (applied by the checks, not here).

Rules (first version; every unclear case gets its own rule name and is counted, never silently guessed):

* spells: game events of type OTB or ``*KICKOFF``. A spell is CONTROL (player and team of the game event) unless it holds
  only RE rows (LOOSE ``rebound_spell``), only CH rows of challengeType FIFTY or A (LOOSE ``duel_spell``), or has no player
  (CH rows: LOOSE ``ch_null_player``; other: UNKNOWN ``null_player_spell``). Overlapping spells: the later one wins.
* between two spells of a period, by the last possession event of the earlier spell: PA/CR/SH -> TRAVELLING (team and
  player that played it); CL/RE/CH, TC with touchOutcomeType P/O/C (C = Challenge, PFF spec p.24) and BC with
  ballCarryOutcome L -> LOOSE (last toucher's team); TC with touchOutcomeType R and BC with ballCarryOutcome R -> CONTROL
  by the same player continues; a spell without any possession event -> UNKNOWN ``gap_after_none`` (not scored); any other
  case -> UNKNOWN ``gap_after_<type>``.
* DEAD (overrides everything): OUT eventTime -> start of the next spell (same period); before the first spell of a period;
  END eventTime -> end of period; ``restart_setup``: a spell whose setpieceType is not O is DEAD from its start until its
  first release (PA/CR/SH/CL) eventTime, except a drop ball (D), which is live from the spell start (Laws of the Game:
  in play once it touches the ground). SUB/ON/OFF/FOUL/G rows never start or end anything.
* score_mask False: frames of spells with videoMissing, frames of spells with a nonEvent / initialNonEvent row, UNKNOWN,
  frames outside a period.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEAD, CONTROL, TRAVELLING, LOOSE, UNKNOWN = 0, 1, 2, 3, 4
STATE_NAMES = ("DEAD", "CONTROL", "TRAVELLING", "LOOSE", "UNKNOWN")
TOLERANCE_S = 0.3                     # label-vs-tracking checks and step D scoring (+-9 frames at 29.97 fps)
RELEASES = ("PA", "CR", "SH", "CL")
SPELL_TYPES = ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF")


@dataclass
class Row:
    idx: int                      # source row index
    ge: int | None                # gameEventId
    pe: int | None                # possessionEventId
    get: str | None               # gameEventType
    pet: str | None               # possessionEventType
    period: int | None
    st: float | None
    et: float | None
    en: float | None
    team: int | None
    player: int | None
    setpiece: str | None = None
    out_type: str | None = None
    video_missing: bool = False
    non_event: bool = False
    initial_non_event: bool = False
    challenge_type: str | None = None
    carry_outcome: str | None = None
    touch_outcome: str | None = None


@dataclass
class Spell:
    ge: int
    period: int
    i0: int
    i1: int
    rows: list
    player: int | None
    team: int | None
    setpiece: str | None
    state: int = CONTROL
    rule: str = "otb_spell"
    span_source: str = "overlay"
    release_row: Row | None = None
    release_frame: int | None = None

    @property
    def last_pet(self) -> str | None:
        pets = [r for r in self.rows if r.pet]
        return pets[-1].pet if pets else None

    @property
    def last_row(self) -> Row:
        pets = [r for r in self.rows if r.pet]
        return pets[-1] if pets else self.rows[-1]

    @property
    def masked(self) -> bool:
        return any(r.video_missing or r.non_event or r.initial_non_event for r in self.rows)


@dataclass
class Labels:
    state: np.ndarray
    player: np.ndarray
    team: np.ndarray
    score_mask: np.ndarray
    src_ge: np.ndarray
    src_pe: np.ndarray
    rule: np.ndarray               # index into rule_names
    rule_names: list
    spells: list
    log: dict = field(default_factory=dict)   # counts and examples of every special case


def _key(r: Row):
    return (r.et if r.et is not None else (r.st if r.st is not None else 1e12), r.idx)


def build_spells(rows: list[Row], spans: dict, row_frames: dict, log: dict) -> list[Spell]:
    """Group rows into spells. spans: ge -> (i0, i1) from the tracking overlay; row_frames: row idx -> {'start','event','end'}."""
    by_ge: dict[int, list[Row]] = {}
    for r in rows:
        if r.get in SPELL_TYPES and r.ge is not None and r.period is not None:
            by_ge.setdefault(r.ge, []).append(r)
    spells = []
    for ge, rs in by_ge.items():
        rs.sort(key=_key)
        if ge in spans:
            i0, i1 = spans[ge]
            src = "overlay"
        else:
            f0 = [row_frames.get(r.idx, {}).get("start") for r in rs]
            f1 = [row_frames.get(r.idx, {}).get("end") for r in rs]
            f0, f1 = [f for f in f0 if f is not None], [f for f in f1 if f is not None]
            if not f0:
                log.setdefault("spell_no_frames", []).append(ge)
                continue
            i0, i1 = min(f0), max(f1) if f1 else max(f0)
            src = "aligned"
        first = rs[0]
        sp = Spell(ge, first.period, int(i0), int(i1), rs, first.player, first.team, first.setpiece, span_source=src)
        pets = [r.pet for r in rs if r.pet]
        if first.player is None:
            if pets and all(p == "CH" for p in pets):
                sp.state, sp.rule = LOOSE, "ch_null_player"
            else:
                sp.state, sp.rule = UNKNOWN, "null_player_spell"
        elif pets and all(p == "RE" for p in pets):
            sp.state, sp.rule = LOOSE, "rebound_spell"
        elif pets and all(p == "CH" and r.challenge_type in ("FIFTY", "A") for r in rs for p in [r.pet] if p):
            sp.state, sp.rule = LOOSE, "duel_spell"
        elif not pets:
            sp.rule = "otb_spell_no_possession_event"
        rel = next((r for r in rs if r.pet in RELEASES), None)
        if rel is not None:
            sp.release_row = rel
            sp.release_frame = row_frames.get(rel.idx, {}).get("event")
        spells.append(sp)
    spells.sort(key=lambda s: (s.period, s.i0, s.rows[0].st if s.rows[0].st is not None else 0.0))
    return spells


def label_match(t: np.ndarray, period: np.ndarray, rows: list[Row], spans: dict, row_frames: dict) -> Labels:
    n = len(t)
    period = np.asarray(period)
    rules: list[str] = []

    def rid(name: str) -> int:
        if name not in rules:
            rules.append(name)
        return rules.index(name)

    state = np.full(n, UNKNOWN, np.int8)
    player = np.full(n, -1, np.int64)
    team = np.full(n, -1, np.int64)
    mask = np.ones(n, bool)
    src_ge = np.full(n, -1, np.int64)
    src_pe = np.full(n, -1, np.int64)
    rule = np.full(n, rid("no_rule"), np.int16)
    log: dict = {}

    def paint(a: int, b: int, st: int, name: str, pl=None, tm=None, ge=None, pe=None):
        if b < a:
            return
        state[a:b + 1] = st
        player[a:b + 1] = -1 if pl is None else pl
        team[a:b + 1] = -1 if tm is None else tm
        src_ge[a:b + 1] = -1 if ge is None else ge
        src_pe[a:b + 1] = -1 if pe is None else pe
        rule[a:b + 1] = rid(name)

    spells = build_spells(rows, spans, row_frames, log)
    out_period = period <= 0
    paint_idx = np.flatnonzero(out_period)
    for i in paint_idx:
        rule[i] = rid("outside_period")
    mask[out_period] = False

    for p in sorted(set(int(x) for x in np.unique(period) if x > 0)):
        fr = np.flatnonzero(period == p)
        f_first, f_last = int(fr[0]), int(fr[-1])
        sps = [s for s in spells if s.period == p]
        # gaps between consecutive spells (by start), decided by the earlier spell's last possession event
        for a, b in zip(sps, sps[1:] + [None]):
            g0, g1 = a.i1 + 1, (b.i0 - 1 if b is not None else f_last)
            last = a.last_pet
            lr = a.last_row
            if last in ("PA", "CR", "SH"):
                paint(g0, g1, TRAVELLING, f"travel_after_{last}", a.player, a.team, a.ge, lr.pe)
            elif (last == "TC" and lr.touch_outcome == "R") or (last == "BC" and lr.carry_outcome == "R"):
                paint(g0, g1, CONTROL, f"control_after_{last}_retained", a.player, a.team, a.ge, lr.pe)
            elif (last in ("CL", "RE", "CH") or (last == "TC" and lr.touch_outcome in ("P", "O", "C"))
                  or (last == "BC" and lr.carry_outcome == "L")):
                name = "loose_after_BC_lost" if last == "BC" else f"loose_after_{last}"
                paint(g0, g1, LOOSE, name, None, lr.team if lr.team is not None else a.team, a.ge, lr.pe)
            else:
                extra = {"BC": lr.carry_outcome, "TC": lr.touch_outcome}.get(last)
                name = (f"gap_after_{last}" + (f"_{extra}" if last in ("BC", "TC") else "")) if last else "gap_after_none"
                paint(g0, g1, UNKNOWN, name, None, a.team, a.ge, lr.pe)
        # spells: later wins on overlap
        reach = -1
        for s in sps:
            if s.i0 <= reach:
                log.setdefault("overlaps", []).append((s.ge, reach - s.i0 + 1))
            pl = s.player if s.state == CONTROL else None
            tm = s.team if s.team is not None else None
            paint(s.i0, s.i1, s.state, s.rule, pl, tm, s.ge, s.last_row.pe)
            reach = max(reach, s.i1)
            if s.masked:
                mask[s.i0:s.i1 + 1] = False
        # DEAD (overrides): before the first spell, OUT -> next spell start, END -> period end, restart setup
        if sps:
            paint(f_first, sps[0].i0 - 1, DEAD, "before_kickoff")
        else:
            paint(f_first, f_last, DEAD, "period_without_spells")
        prow = sorted((r for r in rows if r.period == p and r.et is not None), key=_key)
        for r in prow:
            if r.get == "OUT":
                f = row_frames.get(r.idx, {}).get("event")
                if f is None:
                    log.setdefault("out_unaligned", []).append(r.ge)
                    continue
                nxt = next((s for s in sps if s.rows[0].st is not None and s.rows[0].st >= r.et), None)
                end = nxt.i0 - 1 if nxt is not None else f_last
                paint(f, end, DEAD, f"out_{r.out_type}", ge=r.ge)
                if nxt is not None and nxt.setpiece in (None, "O"):
                    log.setdefault("stoppage_next_no_setpiece", []).append((r.ge, nxt.ge, nxt.setpiece))
                if nxt is None:
                    log.setdefault("out_without_restart", []).append(r.ge)
            elif r.get == "END":
                f = row_frames.get(r.idx, {}).get("event")
                if f is not None:
                    paint(f, f_last, DEAD, "after_end", ge=r.ge)
        for s in sps:
            if s.setpiece in (None, "O"):
                continue
            if s.setpiece == "D":                       # drop ball: live from the spell start
                log.setdefault("drop_ball_live", []).append(s.ge)
                continue
            if s.release_frame is not None and s.release_frame > s.i0:
                paint(s.i0, s.release_frame - 1, DEAD, f"restart_setup_{s.setpiece}", ge=s.ge, pe=s.release_row.pe)
                log.setdefault("restart_setup_s", []).append((s.setpiece, s.release_row.et - s.rows[0].st, s.ge))
            elif s.release_frame is None:
                log.setdefault("restart_without_release", []).append((s.ge, s.setpiece, s.last_pet))
            else:
                log.setdefault("restart_setup_s", []).append((s.setpiece, 0.0, s.ge))
    mask[state == UNKNOWN] = False
    return Labels(state, player, team, mask, src_ge, src_pe, rule, rules, spells, log)
