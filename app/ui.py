"""
UI layer — presentation only
============================
Reusable visual components for the Streamlit app. This module is deliberately
*pure presentation*: it reads the pipeline result dict, it never mutates it,
never calls the engine, never touches session state, and never changes a value
— only how a value is displayed.

Two rules it enforces for the whole app:

1. No internal identifier ever reaches the screen. `super_automatic`,
   `grind_too_fine`, `water_temp` are engine vocabulary; the user sees
   "Super automatic", "Grind too fine", "Water temperature"
   (see :func:`humanize_label`).
2. Action before science. Structured engine output (diagnosis → next
   adjustment → plan) is rendered as scannable cards first; the model's long
   prose follows, with its explanatory / technical sections folded into
   expanders so the first screen answers "what do I change next?".

The visual tokens live in ``.streamlit/config.toml`` (native theme) and
``app/assets/styles.css`` (thin layer). Nothing here hardcodes a color.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
BANNER_PATH = Path(__file__).parent / "BandeauCoffee.png"


# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

def inject_css() -> None:
    """Load the stylesheet. ``st.html`` with a .css path is the native route:
    Streamlit wraps it in <style> and sends style-only content to the event
    container, so it costs no layout space (unlike an st.markdown block)."""
    if CSS_PATH.exists():
        st.html(CSS_PATH)


# ------------------------------------------------------------------
# Label humanisation
# ------------------------------------------------------------------

_LABEL_OVERRIDES = {
    "v60": "V60",
    "aeropress": "AeroPress",
    "french press": "French press",
    "moka": "Moka pot",
    "super automatic": "Super automatic",
    "semi automatic": "Semi automatic",
    "manual lever": "Manual lever",
    "nespresso": "Nespresso",
    "tds": "TDS",
    "ey": "Extraction yield",
    "sca": "SCA",
}

_PARAM_LABELS = {
    "grind_size": "Grind size",
    "water_temp": "Water temperature",
    "dose": "Dose",
    "time": "Extraction time",
    "freshness": "Bean freshness",
    "maintenance": "Maintenance",
    "tamping": "Tamping",
    "distribution": "Distribution",
    "technique": "Pour technique",
    "pour_technique": "Pour technique",
    "heat_level": "Heat level",
    "fill_level": "Fill level",
    "temperature_setting": "Temperature setting",
    "pressure": "Pressure",
    "bloom_time": "Bloom time",
    "steep_time": "Steep time",
    "coffee_strength": "Coffee strength",
    "water_amount": "Water amount",
    "pre_infusion": "Pre-infusion",
    "machine_cleaning": "Machine cleaning",
}

# Engine verdicts → user-facing wording. Display only: the stored value and
# every downstream metric keep the original token.
VERDICT_LABELS = {
    "pass": "Passed",
    "warn": "Needs review",
    "fail": "Failed",
    "blocked": "Blocked",
}

# Where to read the user's *current* value for a given intervention parameter,
# and the unit to render it with.
_PARAM_CURRENT_FIELD = {
    "grind_size": ("grind_size", ""),
    "water_temp": ("water_temp_celsius", "°C"),
    "dose": ("dose_grams", " g"),
    "time": ("extraction_time_seconds", " s"),
}


def humanize_label(value: object, default: str = "Unknown") -> str:
    """`super_automatic` → "Super automatic". Never let an internal token,
    snake_case key or slug reach the screen."""
    if value is None:
        return default
    text = str(value).replace("_", " ").replace("-", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return default
    lowered = text.lower()
    if lowered in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[lowered]
    return text[0].upper() + text[1:]


def parameter_label(parameter: object) -> str:
    """Intervention.parameter → a human parameter name."""
    if parameter is None:
        return "Adjustment"
    return _PARAM_LABELS.get(str(parameter).strip().lower(), humanize_label(parameter, "Adjustment"))


def _format_number(value: object) -> str:
    """22.0 → "22"; 92.5 → "92.5"; anything else → str()."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else f"{value:g}"
    return str(value)


def _e(value: object) -> str:
    """Escape for HTML interpolation — every dynamic value goes through this."""
    return html.escape(str(value), quote=True)


# ------------------------------------------------------------------
# Primitives
# ------------------------------------------------------------------

def banner() -> None:
    """Full-width header image, rendered above :func:`page_header`.

    Goes through ``st.image`` rather than an inline base64 data URI inside
    ``st.html``: the file is ~1.6 MB and Streamlit re-executes the whole script
    on every rerun, so a data URI would re-send it on each keystroke-level
    interaction. ``st.image`` hands the file to the media file manager, which
    serves it once from a stable, browser-cached URL. (The other route — a
    plain ``<img src="app/...">`` — would 404: static serving is off in
    ``.streamlit/config.toml``.)

    The keyed container is the CSS hook: ``st.container(key="hb-banner")``
    renders a ``.st-key-hb-banner`` class, a documented user-key selector
    rather than a Streamlit internal, so the radius/spacing live in the
    stylesheet like every other component here.

    Missing file → renders nothing. The banner is decoration; it must never
    take the app down on a checkout where the asset wasn't pulled.
    """
    if not BANNER_PATH.exists():
        return
    with st.container(key="hb-banner"):
        st.image(str(BANNER_PATH), use_container_width=True)


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    """Small identity line, a measured H1, and a quiet subtitle."""
    sub = f'<p class="hb-hero__subtitle">{_e(subtitle)}</p>' if subtitle else ""
    st.html(
        '<div class="hb-hero">'
        f'<p class="hb-eyebrow">{_e(eyebrow)}</p>'
        f'<h1 class="hb-hero__title">{_e(title)}</h1>'
        f"{sub}"
        "</div>"
    )


def section_label(text: str) -> None:
    st.html(f'<p class="hb-section-label">{_e(text)}</p>')


def rule() -> None:
    st.html('<hr class="hb-rule">')


def metric_cards(cards: list[dict]) -> None:
    """A responsive row of compact metric cards.

    One CSS-grid block rather than st.columns: the grid auto-fits, so the
    cards stack on a 390px screen without any width math or breakpoint list.

    Each card: {"label", "value", "sub" (optional), "ok" (optional bool)}.
    """
    if not cards:
        return
    items = []
    for card in cards:
        tone = " hb-card__value--ok" if card.get("ok") else ""
        sub = f'<div class="hb-card__sub">{_e(card["sub"])}</div>' if card.get("sub") else ""
        items.append(
            '<div class="hb-card">'
            f'<div class="hb-card__label">{_e(card.get("label", ""))}</div>'
            f'<div class="hb-card__value{tone}">{_e(card.get("value", "—"))}</div>'
            f"{sub}"
            "</div>"
        )
    st.html(f'<div class="hb-cards">{"".join(items)}</div>')


def param_card(
    name: str,
    target: str,
    current: str | None = None,
    delta: str | None = None,
    note: str | None = None,
) -> str:
    """HTML for one Current → Target brewing metric (not a spreadsheet row).

    Returns the fragment so several cards can share one grid; use
    :func:`param_grid` to render them.
    """
    if current:
        cells = (
            '<div class="hb-param__cell">'
            f'<div class="hb-param__value hb-param__value--from">{_e(current)}</div>'
            '<div class="hb-param__cap">Current</div></div>'
            '<div class="hb-param__arrow">&rarr;</div>'
            '<div class="hb-param__cell">'
            f'<div class="hb-param__value hb-param__value--to">{_e(target)}</div>'
            '<div class="hb-param__cap">Target</div></div>'
        )
    else:
        cells = (
            '<div class="hb-param__cell">'
            f'<div class="hb-param__value hb-param__value--to">{_e(target)}</div>'
            '<div class="hb-param__cap">Target</div></div>'
        )
    chip = f'<div class="hb-param__delta">{_e(delta)}</div>' if delta else ""
    hint = f'<div class="hb-param__note">{_e(note)}</div>' if note else ""
    return (
        '<div class="hb-param">'
        f'<div class="hb-param__name">{_e(name)}</div>'
        f'<div class="hb-param__row">{cells}</div>'
        f"{chip}{hint}"
        "</div>"
    )


def param_grid(cards: list[str]) -> None:
    if cards:
        st.html(f'<div class="hb-params">{"".join(cards)}</div>')


def step_heading(number: object, title: str) -> None:
    """Quiet two-digit number, action-oriented title — replaces the oversized
    "2. Adjust Grind Coarseness" headings."""
    num = ""
    if number is not None:
        try:
            num = f'<span class="hb-inline-step__num">{int(number):02d}</span>'
        except (TypeError, ValueError):
            num = f'<span class="hb-inline-step__num">{_e(number)}</span>'
    st.html(
        '<div class="hb-inline-step">'
        f"{num}"
        f'<span class="hb-inline-step__title">{_e(title)}</span>'
        "</div>"
    )


def checklist(items: list[str]) -> None:
    if not items:
        return
    lis = "".join(f"<li>{_e(item)}</li>" for item in items)
    st.html(f'<ul class="hb-checklist">{lis}</ul>')


def brand(name_lines: list[str], tagline: str = "", mark: str = "☕") -> None:
    """Sidebar identity — one compact block instead of a title + caption stack."""
    label = "<br>".join(_e(line) for line in name_lines)
    tag = f'<p class="hb-brand__tag">{_e(tagline)}</p>' if tagline else ""
    st.html(
        '<div class="hb-brand">'
        f'<span class="hb-brand__mark">{_e(mark)}</span>'
        f'<span class="hb-brand__name">{label}</span>'
        "</div>"
        f"{tag}"
    )


def quota_note(text: str) -> None:
    """Deliberately understated: a budget is context, not headline information."""
    st.html(f'<p class="hb-quota">{_e(text)}</p>')


# ------------------------------------------------------------------
# Diagnosis / adjustment / plan — built from engine output
# ------------------------------------------------------------------

_TOO_RE = re.compile(r"^(?!too\b)(.+?)\s+too\s+(.+)$", re.I)


def _claim_sentence(hypothesis: str) -> str:
    """Turn a root-cause slug into a readable claim.

    "grind_too_fine" → "Your grind is probably too fine."
    Anything that does not fit the "<thing> too <quality>" shape falls back to
    a neutral phrasing — we rewrite the *label*, never the diagnosis.
    """
    label = humanize_label(hypothesis, "")
    if not label:
        return "Most likely cause identified below."
    match = _TOO_RE.match(label)
    if match:
        subject, quality = match.group(1).strip().lower(), match.group(2).strip().lower()
        return f"Your {subject} is probably too {quality}."
    return f"Most likely cause: {label.lower()}."


def render_diagnosis(diagnostic: dict) -> None:
    """DIAGNOSIS card: the claim, the confidence, and why it follows."""
    root_causes = diagnostic.get("root_causes") or []
    if not root_causes:
        return
    primary = root_causes[0] or {}
    confidence = diagnostic.get("diagnostic_confidence") or 0.0
    try:
        pct = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        pct = 0.0

    evidence = (primary.get("evidence") or "").strip()
    why = f'<p class="hb-diagnosis__why">{_e(evidence)}</p>' if evidence else ""

    st.html(
        '<div class="hb-diagnosis">'
        '<div class="hb-diagnosis__head">'
        '<span class="hb-eyebrow" style="margin:0">Diagnosis</span>'
        f'<span class="hb-diagnosis__confidence"><b>{pct:.0%}</b> confidence</span>'
        "</div>"
        f'<p class="hb-diagnosis__claim">{_e(_claim_sentence(primary.get("hypothesis", "")))}</p>'
        f"{why}"
        f'<div class="hb-meter"><div class="hb-meter__fill" style="width:{pct:.0%}"></div></div>'
        "</div>"
    )


def _current_value(intervention: dict, context: dict) -> str | None:
    """The user's current value for this parameter, if they told us. Read-only
    lookup into the extracted context — nothing is inferred or computed."""
    field = _PARAM_CURRENT_FIELD.get(str(intervention.get("parameter") or "").lower())
    if not field:
        return None
    key, unit = field
    value = context.get(key)
    if value in (None, ""):
        return None
    return f"{_format_number(value)}{unit}"


def _target_phrase(intervention: dict) -> str | None:
    """The requested change, in words — "1 notch coarser". Deliberately not a
    computed number: the engine gives a direction and a magnitude, and
    inventing "3 → 4" would be fabricating a value it never produced."""
    direction = intervention.get("direction")
    magnitude = intervention.get("magnitude")
    if magnitude and direction:
        return f"{magnitude} {direction}"
    if direction:
        return humanize_label(direction, "")
    if magnitude:
        return str(magnitude)
    return None


def render_next_adjustment(diagnostic: dict, context: dict) -> None:
    """NEXT ADJUSTMENT: the single change to test first."""
    plan = diagnostic.get("intervention_plan") or []
    if not plan:
        return
    first = plan[0] or {}
    action = (first.get("action") or "").strip()
    if not action:
        return

    section_label("Next adjustment")
    st.markdown(f"**{action}**")

    target = _target_phrase(first)
    if target:
        param_grid([
            param_card(
                name=parameter_label(first.get("parameter")),
                target=target,
                current=_current_value(first, context),
                delta=first.get("magnitude") or None,
                note=(first.get("expected_result") or "").strip() or None,
            )
        ])
    elif first.get("expected_result"):
        st.caption(first["expected_result"])


def render_plan(diagnostic: dict) -> None:
    """RECOMMENDED PLAN: numbered steps, number small and secondary."""
    plan = diagnostic.get("intervention_plan") or []
    if len(plan) < 2:
        return

    section_label("Recommended plan")
    rows = []
    for index, step in enumerate(plan, start=1):
        action = (step.get("action") or "").strip()
        if not action:
            continue
        try:
            number = int(step.get("step") or index)
        except (TypeError, ValueError):
            number = index
        expected = (step.get("expected_result") or "").strip()
        meta = f'<div class="hb-step__meta">{_e(expected)}</div>' if expected else ""
        rows.append(
            '<div class="hb-step">'
            f'<div class="hb-step__num">{number:02d}</div>'
            '<div class="hb-step__body">'
            f'<div class="hb-step__title">{_e(action)}</div>'
            f"{meta}"
            "</div></div>"
        )
    if rows:
        st.html(f'<div class="hb-steps">{"".join(rows)}</div>')


def render_validation(diagnostic: dict) -> None:
    """How to confirm the fix worked — taste first, instruments in a second
    level (see :func:`_split_advanced`)."""
    plan = diagnostic.get("intervention_plan") or []
    tests, seen = [], set()
    for step in plan:
        test = (step.get("validation_test") or "").strip()
        if test and test.lower() not in seen:
            seen.add(test.lower())
            tests.append(test)
    if not tests:
        return

    basic = [t for t in tests if not _ADVANCED_RE.search(t)]
    advanced = [t for t in tests if _ADVANCED_RE.search(t)]

    section_label("Brew again and taste")
    if basic:
        st.caption("You're looking for:")
        checklist(basic)
    if advanced:
        with st.expander("Advanced validation"):
            checklist(advanced)


# ------------------------------------------------------------------
# Model prose — reshaped, never rewritten
# ------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")
# Models often emit a bold line instead of an ATX heading — treat a short,
# fully-bold line as a heading too.
_BOLD_HEADING_RE = re.compile(r"^\s*\*\*(.{2,80}?)\*\*\s*:?\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_STEP_TITLE_RE = re.compile(r"^(?:step\s*)?(\d{1,2})\s*[.)\-–:]\s+(.+)$", re.I)

_SCIENCE_RE = re.compile(
    r"\b(why|science|scientific|chemistry|theory|mechanism|background|"
    r"deep dive|technical (analysis|detail|note)|the science)\b", re.I
)
_VALIDATION_RE = re.compile(r"\b(validation|validate|verify|verification|how to (check|test))\b", re.I)
_ADVANCED_RE = re.compile(
    r"\b(tds|refractomet\w*|extraction yield|brix|vst|atago|"
    r"digital scale to 0\.1|puck screen)\b", re.I
)

_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_PARAM_LINE_RE = re.compile(
    r"^\s*[-*+]?\s*\**(current|now|before|target|goal|after|change|delta|adjust(?:ment)?)\**"
    r"\s*[:：]\s*\**(.+?)\**\s*$", re.I
)


def _split_sections(markdown_text: str) -> tuple[str, list[dict]]:
    """Split prose into (preamble, [{title, body}]) on markdown headings.
    Fenced code is never treated as a heading."""
    preamble: list[str] = []
    sections: list[dict] = []
    current: dict | None = None
    in_fence = False

    for line in markdown_text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None
        if not in_fence:
            match = _HEADING_RE.match(line)
            if match:
                heading = match.group(2).strip()
            else:
                bold = _BOLD_HEADING_RE.match(line)
                # Only a *title-like* bold line, not a bolded sentence.
                if bold and not bold.group(1).rstrip().endswith((".", "!", "?")):
                    heading = bold.group(1).strip().rstrip(":")
        if heading:
            current = {"title": heading, "body": []}
            sections.append(current)
        elif current is None:
            preamble.append(line)
        else:
            current["body"].append(line)

    return "\n".join(preamble).strip(), sections


def _step_parts(title: str) -> tuple[int | None, str]:
    """"2. Adjust Grind Coarseness" → (2, "Adjust Grind Coarseness")."""
    match = _STEP_TITLE_RE.match(title.strip())
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, title.strip()


def _split_advanced(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate instrument-grade lines (TDS, refractometer…) from taste-level
    ones. Nothing is dropped — the advanced lines move to a second level."""
    basic, advanced = [], []
    for line in lines:
        (advanced if _ADVANCED_RE.search(line) else basic).append(line)
    return basic, advanced


def _table_rows(block: list[str]) -> list[list[str]]:
    rows = []
    for line in block:
        if _TABLE_SEP_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def _params_from_table(block: list[str]) -> list[str] | None:
    """A table whose columns are essentially "parameter / current / target" is
    a recipe change, not data — render it as brewing metrics. Any other table
    is left alone: real tables earn their shape."""
    rows = _table_rows(block)
    if len(rows) < 2:
        return None
    header = [c.lower() for c in rows[0]]
    if len(header) < 2:
        return None

    def _find(*keys: str) -> int | None:
        for index, cell in enumerate(header):
            if any(key in cell for key in keys):
                return index
        return None

    current_col = _find("current", "now", "before", "from")
    target_col = _find("target", "goal", "after", "recommend", "to ")
    if current_col is None or target_col is None or current_col == target_col:
        return None
    delta_col = _find("change", "delta", "diff")
    name_col = next(
        (i for i in range(len(header)) if i not in {current_col, target_col, delta_col}),
        None,
    )

    cards = []
    for row in rows[1:]:
        if len(row) <= max(current_col, target_col):
            continue
        current, target = row[current_col], row[target_col]
        if not current and not target:
            continue
        name = row[name_col] if name_col is not None and name_col < len(row) else "Adjustment"
        delta = row[delta_col] if delta_col is not None and delta_col < len(row) else None
        cards.append(
            param_card(
                name=humanize_label(re.sub(r"\*+", "", name) or "Adjustment", "Adjustment"),
                target=re.sub(r"\*+", "", target) or "—",
                current=re.sub(r"\*+", "", current) or None,
                delta=re.sub(r"\*+", "", delta) if delta else None,
            )
        )
    return cards or None


def _params_from_lines(block: list[str], name: str) -> list[str] | None:
    """The inline shape: "Current: 3" / "Target: 4" on consecutive lines."""
    values: dict[str, str] = {}
    for line in block:
        match = _PARAM_LINE_RE.match(line)
        if not match:
            return None
        values[match.group(1).lower()] = re.sub(r"\*+", "", match.group(2)).strip()
    current = values.get("current") or values.get("now") or values.get("before")
    target = values.get("target") or values.get("goal") or values.get("after")
    if not target:
        return None
    delta = values.get("change") or values.get("delta") or values.get("adjustment") or values.get("adjust")
    return [param_card(name=name, target=target, current=current, delta=delta)]


def _render_body(lines: list[str], context_name: str = "Adjustment") -> None:
    """Render a section body, promoting parameter tables/lines to metric cards
    and leaving everything else to Streamlit's markdown."""
    buffer: list[str] = []

    def flush_markdown() -> None:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            st.markdown(text)

    index = 0
    while index < len(lines):
        line = lines[index]

        # Markdown table?
        if line.strip().startswith("|") and index + 1 < len(lines) and _TABLE_SEP_RE.match(lines[index + 1]):
            end = index
            while end < len(lines) and lines[end].strip().startswith("|"):
                end += 1
            block = lines[index:end]
            cards = _params_from_table(block)
            flush_markdown()
            if cards:
                param_grid(cards)
            else:
                st.markdown("\n".join(block))
            index = end
            continue

        # Consecutive "Current: / Target:" lines?
        if _PARAM_LINE_RE.match(line):
            end = index
            while end < len(lines) and _PARAM_LINE_RE.match(lines[end]):
                end += 1
            cards = _params_from_lines(lines[index:end], context_name)
            flush_markdown()
            if cards:
                param_grid(cards)
            else:
                st.markdown("\n".join(lines[index:end]))
            index = end
            continue

        buffer.append(line)
        index += 1

    flush_markdown()


def _prose_has_validation(markdown_text: str) -> bool:
    """True when the generated text already carries its own validation section."""
    try:
        _, sections = _split_sections(markdown_text or "")
    except Exception:
        return False
    return any(_VALIDATION_RE.search(section["title"]) for section in sections)


def render_prose(markdown_text: str) -> None:
    """Render the model's coaching text with a visual hierarchy applied.

    The text itself is never rewritten — only re-laid-out: numbered headings
    become small step markers, "current/target" tables become brewing metrics,
    and the explanatory / instrument-level material moves into expanders so
    the practical fix stays on the first screen.
    """
    text = (markdown_text or "").strip()
    if not text:
        return
    try:
        preamble, sections = _split_sections(text)
    except Exception:  # presentation must never break the answer
        st.markdown(text)
        return

    if not sections:
        st.markdown(text)
        return

    if preamble:
        _render_body(preamble.splitlines())

    for section in sections:
        body = section["body"]
        number, title = _step_parts(section["title"])

        if _VALIDATION_RE.search(title):
            step_heading(None, "Brew again and taste")
            basic, advanced = _split_advanced(body)
            if any(line.strip() for line in basic):
                _render_body(basic)
            if any(line.strip() for line in advanced):
                with st.expander("Advanced validation"):
                    _render_body(advanced)
            continue

        # Fold the science into a second level — but only when there is a
        # first level left to read, and only when it is actually long. A short
        # aside, or an answer made of one single section, stays in the open:
        # hiding it would remove information rather than rank it.
        is_long = len(" ".join(body).split()) >= 40
        if _SCIENCE_RE.search(title) and len(sections) > 1 and is_long:
            label = title if title.endswith("?") else f"Why this works — {title}"
            with st.expander(label):
                _render_body(body)
            continue

        step_heading(number, title)
        _render_body(body, context_name=title)


# ------------------------------------------------------------------
# Top-level composition
# ------------------------------------------------------------------

def render_coaching_answer(result: dict, coaching_text: str) -> None:
    """The coach's reply: scannable structure first, then the full prose.

    First screen answers, in order: what is probably wrong, what to change
    next, and which targets to aim for. The model's explanation follows.
    """
    diagnostic = result.get("diagnostic") or {}
    context = result.get("context") or {}
    has_structure = bool(diagnostic.get("root_causes") or diagnostic.get("intervention_plan"))

    if has_structure:
        render_diagnosis(diagnostic)
        render_next_adjustment(diagnostic, context)
        render_plan(diagnostic)
        # The coach's own text usually closes on a validation step. Only fall
        # back to the engine's validation_test fields when it doesn't, so the
        # reader never gets the same section twice.
        if not _prose_has_validation(coaching_text):
            render_validation(diagnostic)
        rule()

    render_prose(coaching_text)


def render_session_meta(result: dict) -> None:
    """The closing metric row: machine, confidence, quality check."""
    context = result.get("context") or {}
    diagnostic = result.get("diagnostic") or {}
    evaluation = result.get("evaluation") or {}

    cards: list[dict] = [
        {"label": "Machine", "value": humanize_label(context.get("machine_type"), "Unknown")}
    ]

    if diagnostic.get("root_causes"):
        confidence = diagnostic.get("diagnostic_confidence") or 0.0
        try:
            cards.append({"label": "Confidence", "value": f"{float(confidence):.0%}"})
        except (TypeError, ValueError):
            pass

    verdict = evaluation.get("overall_verdict")
    if verdict:
        passed = str(verdict).lower() == "pass"
        cards.append({
            "label": "Quality check",
            "value": ("✓ " if passed else "") + VERDICT_LABELS.get(str(verdict).lower(), humanize_label(verdict)),
            "ok": passed,
        })

    metric_cards(cards)
