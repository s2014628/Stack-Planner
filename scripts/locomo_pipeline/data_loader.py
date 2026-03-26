import json
import os
import random
import re
from typing import List, Dict, Any, Optional, Tuple

# LoCoMo Category 5 = 对抗性问题（adversarial）：答案不在对话中，
# 模型应回答 "no information available" 或 "not mentioned"。
# 原始数据中 category 5 没有 answer 字段，只有 adversarial_answer（诱导性错误答案）。
# 参考 LoCoMo 官方评估方式，category 5 的问题被格式化为选择题。
CATEGORY_5_GROUND_TRUTH = "no information available"


def load_locomo_data(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_evidence_ref(ref: str) -> Tuple[int, int]:
    """Parse evidence reference like 'D1:3' into (session_num, dialog_num).

    Returns (-1, -1) if the reference cannot be parsed.
    """
    match = re.match(r"D(\d+):(\d+)", ref)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1


def _build_dia_id_index(
    conversation: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Build a mapping from dia_id (e.g. 'D1:3') to its dialog dict + session info."""
    index: Dict[str, Dict[str, Any]] = {}
    session_keys = sorted(
        [
            k
            for k in conversation.keys()
            if k.startswith("session_") and "date_time" not in k
        ],
        key=lambda x: int(x.split("_")[1]),
    )
    for session_key in session_keys:
        session_num = session_key.split("_")[1]
        date_time_key = f"session_{session_num}_date_time"
        date_time = conversation.get(date_time_key, "Unknown date")
        dialogs = conversation[session_key]
        if not isinstance(dialogs, list):
            continue
        for dialog in dialogs:
            dia_id = dialog.get("dia_id", "")
            if dia_id:
                index[dia_id] = {
                    **dialog,
                    "session_num": int(session_num),
                    "session_date_time": date_time,
                }
    return index


def extract_evidence_snippets(
    conversation: Dict[str, Any],
    evidence_refs: List[str],
    context_window: int = 1,
) -> List[Dict[str, Any]]:
    """Extract dialogue snippets for the given evidence references.

    For each evidence ref (e.g. 'D1:3'), returns the matching dialog turn
    together with ``context_window`` turns before and after it so the
    snippet has enough surrounding context.

    Returns a list of dicts, each containing:
      - dia_id, speaker, text, blip_caption (from the evidence turn)
      - session_num, session_date_time
      - context_before / context_after: neighbouring dialog turns
    """
    dia_index = _build_dia_id_index(conversation)

    # Also build per-session ordered lists for context lookup
    session_dialogs: Dict[int, List[Dict[str, Any]]] = {}
    session_keys = sorted(
        [
            k
            for k in conversation.keys()
            if k.startswith("session_") and "date_time" not in k
        ],
        key=lambda x: int(x.split("_")[1]),
    )
    for session_key in session_keys:
        snum = int(session_key.split("_")[1])
        dialogs = conversation[session_key]
        if isinstance(dialogs, list):
            session_dialogs[snum] = dialogs

    snippets: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for ref in evidence_refs:
        if ref in seen_ids:
            continue
        seen_ids.add(ref)

        info = dia_index.get(ref)
        if info is None:
            continue

        snum = info["session_num"]
        dialogs_in_session = session_dialogs.get(snum, [])

        # Find position of the evidence turn in its session
        pos = -1
        for i, d in enumerate(dialogs_in_session):
            if d.get("dia_id") == ref:
                pos = i
                break

        ctx_before = []
        ctx_after = []
        if pos >= 0:
            start = max(0, pos - context_window)
            end = min(len(dialogs_in_session), pos + context_window + 1)
            ctx_before = [
                {
                    "dia_id": d.get("dia_id", ""),
                    "speaker": d.get("speaker", ""),
                    "text": d.get("text", ""),
                }
                for d in dialogs_in_session[start:pos]
            ]
            ctx_after = [
                {
                    "dia_id": d.get("dia_id", ""),
                    "speaker": d.get("speaker", ""),
                    "text": d.get("text", ""),
                }
                for d in dialogs_in_session[pos + 1 : end]
            ]

        snippets.append(
            {
                "dia_id": ref,
                "speaker": info.get("speaker", ""),
                "text": info.get("text", ""),
                "blip_caption": info.get("blip_caption", ""),
                "session_num": snum,
                "session_date_time": info.get("session_date_time", ""),
                "context_before": ctx_before,
                "context_after": ctx_after,
            }
        )

    return snippets


def get_evidence_session_context(
    conv_data: Dict[str, Any],
    evidence_refs: List[str],
) -> Dict[str, Any]:
    """Collect session-level context (observation, summary, events) for sessions
    referenced by the evidence.

    Returns a dict keyed by session number with observation, summary, and events.
    """
    observation = conv_data.get("observation", {})
    session_summary = conv_data.get("session_summary", {})
    event_summary = conv_data.get("event_summary", {})

    session_nums: set = set()
    for ref in evidence_refs:
        snum, _ = parse_evidence_ref(ref)
        if snum > 0:
            session_nums.add(snum)

    context: Dict[str, Any] = {}
    for snum in sorted(session_nums):
        entry: Dict[str, Any] = {"session_num": snum}

        obs_key = f"session_{snum}_observation"
        if obs_key in observation:
            entry["observation"] = observation[obs_key]

        sum_key = f"session_{snum}_summary"
        if sum_key in session_summary:
            entry["summary"] = session_summary[sum_key]

        evt_key = f"events_session_{snum}"
        if evt_key in event_summary:
            entry["events"] = event_summary[evt_key]

        context[str(snum)] = entry

    return context


def format_conversation_history(conversation: Dict[str, Any]) -> str:
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    session_keys = sorted(
        [
            k
            for k in conversation.keys()
            if k.startswith("session_") and "date_time" not in k
        ],
        key=lambda x: int(x.split("_")[1]),
    )

    history_parts = []
    for session_key in session_keys:
        session_num = session_key.split("_")[1]
        date_time_key = f"session_{session_num}_date_time"
        date_time = conversation.get(date_time_key, "Unknown date")

        dialogs = conversation[session_key]
        if not isinstance(dialogs, list):
            continue

        session_text = f"Session {session_num} ({date_time}):\n"
        for dialog in dialogs:
            speaker = dialog.get("speaker", "Unknown")
            text = dialog.get("text", "")
            dia_id = dialog.get("dia_id", "")
            blip_caption = dialog.get("blip_caption", "")

            line = f"  [{dia_id}] {speaker}: {text}"
            if blip_caption:
                line += f" [shared image: {blip_caption}]"
            session_text += line + "\n"

        history_parts.append(session_text)

    return "\n".join(history_parts)


def extract_qa_samples(
    locomo_data: List[Dict[str, Any]],
    categories: Optional[List[int]] = None,
    max_samples_per_conversation: Optional[int] = None,
) -> List[Dict[str, Any]]:
    samples = []

    for conv_data in locomo_data:
        sample_id = conv_data.get("sample_id", "unknown")
        conversation = conv_data.get("conversation", {})
        qa_list = conv_data.get("qa", [])
        observation = conv_data.get("observation", {})
        session_summary = conv_data.get("session_summary", {})
        event_summary = conv_data.get("event_summary", {})

        history = format_conversation_history(conversation)

        count = 0
        for qa_idx, qa in enumerate(qa_list):
            category = qa.get("category", 0)
            if categories and category not in categories:
                continue

            question = qa.get("question", "")
            adversarial_answer = qa.get("adversarial_answer", "")

            if category == 5:
                # Category 5: 对抗性问题，答案不在对话中。
                # 参考 LoCoMo 官方评估 (gpt_utils.py)，格式化为选择题：
                # "Select the correct answer: (a) ... (b) ..."
                # 随机交换选项顺序，避免位置偏差。
                answer = CATEGORY_5_GROUND_TRUTH
                if adversarial_answer:
                    if random.random() < 0.5:
                        question = (
                            f"{question} Select the correct answer: "
                            f"(a) Not mentioned in the conversation "
                            f"(b) {adversarial_answer}"
                        )
                    else:
                        question = (
                            f"{question} Select the correct answer: "
                            f"(a) {adversarial_answer} "
                            f"(b) Not mentioned in the conversation"
                        )
            else:
                answer = qa.get("answer", "")
                if answer is None:
                    answer = ""
                answer = str(answer)

            evidence_refs = qa.get("evidence", [])

            # Extract the actual dialogue snippets for each evidence reference
            evidence_snippets = extract_evidence_snippets(
                conversation, evidence_refs, context_window=1
            )

            # Collect session-level context for evidence sessions
            evidence_session_context = get_evidence_session_context(
                conv_data, evidence_refs
            )

            sample = {
                "sample_id": sample_id,
                "qa_index": qa_idx,
                "category": category,
                "history": history,
                "question": question,
                "ground_truth": answer,
                "adversarial_answer": adversarial_answer if category == 5 else "",
                "evidence": evidence_refs,
                "evidence_snippets": evidence_snippets,
                "evidence_session_context": evidence_session_context,
                "metadata": {
                    "speaker_a": conversation.get("speaker_a", ""),
                    "speaker_b": conversation.get("speaker_b", ""),
                    "total_sessions": len(
                        [
                            k
                            for k in conversation.keys()
                            if k.startswith("session_") and "date_time" not in k
                        ]
                    ),
                    "total_qa": len(qa_list),
                },
            }
            samples.append(sample)
            count += 1

            if max_samples_per_conversation and count >= max_samples_per_conversation:
                break

    return samples


def format_evidence_text(sample: Dict[str, Any]) -> str:
    """Format evidence snippets into a human-readable text block.

    This produces a concise representation of only the evidence-relevant
    dialogue turns (with surrounding context) instead of the full history.
    """
    snippets = sample.get("evidence_snippets", [])
    if not snippets:
        return ""

    parts: List[str] = []
    for snip in snippets:
        header = f"Session {snip['session_num']} ({snip['session_date_time']}):"
        lines = []
        for ctx in snip.get("context_before", []):
            lines.append(f"  [{ctx['dia_id']}] {ctx['speaker']}: {ctx['text']}")
        main_line = f"  [{snip['dia_id']}] {snip['speaker']}: {snip['text']}"
        if snip.get("blip_caption"):
            main_line += f" [shared image: {snip['blip_caption']}]"
        lines.append(f"  >> {main_line.strip()}")
        for ctx in snip.get("context_after", []):
            lines.append(f"  [{ctx['dia_id']}] {ctx['speaker']}: {ctx['text']}")
        parts.append(header + "\n" + "\n".join(lines))

    return "\n\n".join(parts)


def build_user_query(sample: Dict[str, Any]) -> str:
    history = sample["history"]
    question = sample["question"]

    query = (
        f"Below is a conversation history between two people. "
        f"Based on this conversation, please answer the following question.\n\n"
        f"--- Conversation History ---\n{history}\n"
        f"--- End of Conversation ---\n\n"
        f"Question: {question}\n\n"
        f"Please provide a short, concise answer based on the conversation above. "
        f"Answer with exact words from the conversation whenever possible."
    )
    return query


def save_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and preprocess LoCoMo data")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to locomo10.json"
    )
    parser.add_argument("--output-path", type=str, default="./data/locomo_samples.json")
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-per-conv", type=int, default=None)
    args = parser.parse_args()

    data = load_locomo_data(args.data_path)
    samples = extract_qa_samples(
        data, categories=args.categories, max_samples_per_conversation=args.max_per_conv
    )
    save_samples(samples, args.output_path)
    print(f"Extracted {len(samples)} QA samples, saved to {args.output_path}")
