import json
import os
from typing import List, Dict, Any, Optional


def load_locomo_data(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def format_conversation_history(conversation: Dict[str, Any]) -> str:
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    session_keys = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and "date_time" not in k],
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

            answer = qa.get("answer", "")
            if answer is None:
                answer = ""
            answer = str(answer)

            sample = {
                "sample_id": sample_id,
                "qa_index": qa_idx,
                "category": category,
                "history": history,
                "question": qa.get("question", ""),
                "ground_truth": answer,
                "evidence": qa.get("evidence", []),
                "metadata": {
                    "speaker_a": conversation.get("speaker_a", ""),
                    "speaker_b": conversation.get("speaker_b", ""),
                    "total_sessions": len(
                        [k for k in conversation.keys() if k.startswith("session_") and "date_time" not in k]
                    ),
                    "total_qa": len(qa_list),
                },
            }
            samples.append(sample)
            count += 1

            if max_samples_per_conversation and count >= max_samples_per_conversation:
                break

    return samples


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
    parser.add_argument("--data-path", type=str, required=True, help="Path to locomo10.json")
    parser.add_argument("--output-path", type=str, default="./data/locomo_samples.json")
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-per-conv", type=int, default=None)
    args = parser.parse_args()

    data = load_locomo_data(args.data_path)
    samples = extract_qa_samples(data, categories=args.categories, max_samples_per_conversation=args.max_per_conv)
    save_samples(samples, args.output_path)
    print(f"Extracted {len(samples)} QA samples, saved to {args.output_path}")
