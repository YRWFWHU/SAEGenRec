"""SFT data preparation utilities.

In trl >= 0.29, response-only loss masking is handled natively by
``SFTTrainer`` via ``SFTConfig(assistant_only_loss=True)`` for
conversational datasets.  This module provides helper functions to
convert the raw SFT data (instruction/input/output) into the
conversational format that ``SFTTrainer`` expects.
"""

from __future__ import annotations

from datasets import Dataset, Features, Value

MESSAGES_FEATURES = Features(
    {
        "messages": [
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        ],
        "task_type": Value("string"),
    }
)


def convert_to_conversational(raw_dataset: Dataset) -> Dataset:
    """Convert raw SFT dataset to conversational format.

    Transforms each (instruction, input, output) tuple into a list of
    chat messages: ``[{"role": "user", ...}, {"role": "assistant", ...}]``.

    The returned dataset has columns ``messages`` and ``task_type``.
    """

    def _to_messages(row: dict) -> dict:
        user_content = row["instruction"]
        if row["input"]:
            user_content = f"{user_content}\n{row['input']}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["output"]},
        ]
        return {"messages": messages, "task_type": row["task_type"]}

    return raw_dataset.map(
        _to_messages,
        remove_columns=["instruction", "input", "output"],
        desc="Converting to conversational format",
    )
