"""HuggingFace Dataset Feature schemas for all pipeline stages."""

from datasets import Features, Sequence, Value

INTERACTIONS_FEATURES = Features(
    {
        "user_id": Value("string"),
        "item_id": Value("string"),
        "timestamp": Value("int64"),
        "rating": Value("float32"),
        "review_text": Value("string"),
        "review_summary": Value("string"),
    }
)

USER_SEQUENCES_FEATURES = Features(
    {
        "user_id": Value("int32"),
        "item_ids": Sequence(Value("int32")),
        "timestamps": Sequence(Value("int64")),
        "ratings": Sequence(Value("float32")),
        "review_texts": Sequence(Value("string")),
        "review_summaries": Sequence(Value("string")),
    }
)

ITEM_METADATA_FEATURES = Features(
    {
        "item_id": Value("string"),
        "title": Value("string"),
        "brand": Value("string"),
        "categories": Sequence(Value("string")),
        "description": Value("string"),
        "price": Value("float32"),
        "image_url": Value("string"),
    }
)

ID_MAP_FEATURES = Features(
    {
        "original_id": Value("string"),
        "mapped_id": Value("int32"),
    }
)

TRAINING_SAMPLE_FEATURES = Features(
    {
        "user_id": Value("int32"),
        "history_item_ids": Sequence(Value("int32")),
        "history_item_tokens": Sequence(Sequence(Value("int32"))),
        "history_item_titles": Sequence(Value("string")),
        "target_item_id": Value("int32"),
        "target_item_tokens": Sequence(Value("int32")),
        "target_item_title": Value("string"),
    }
)

INTERIM_SAMPLE_FEATURES = Features(
    {
        "user_id": Value("int32"),
        "history_item_ids": Sequence(Value("int32")),
        "history_item_titles": Sequence(Value("string")),
        "target_item_id": Value("int32"),
        "target_item_title": Value("string"),
    }
)

NEGATIVE_SAMPLE_FEATURES = Features(
    {
        "user_id": Value("int32"),
        "history_item_ids": Sequence(Value("int32")),
        "history_item_titles": Sequence(Value("string")),
        "target_item_id": Value("int32"),
        "target_item_title": Value("string"),
        "negative_item_ids": Sequence(Value("int32")),
        "negative_item_titles": Sequence(Value("string")),
    }
)

TEXT_EMBEDDING_FEATURES = Features(
    {
        "item_id": Value("int32"),
        "embedding": Sequence(Value("float32")),
    }
)

SEMANTIC_EMBEDDING_FEATURES = Features(
    {
        "item_id": Value("int32"),
        "embedding": Sequence(Value("float32")),
    }
)

COLLABORATIVE_EMBEDDING_FEATURES = Features(
    {
        "item_id": Value("int32"),
        "embedding": Sequence(Value("float32")),
    }
)

SID_MAP_FEATURES = Features(
    {
        "item_id": Value("int32"),
        "codes": Sequence(Value("int32")),
        "sid_tokens": Value("string"),
    }
)

SFT_FEATURES = Features(
    {
        "task_type": Value("string"),
        "instruction": Value("string"),
        "input": Value("string"),
        "output": Value("string"),
    }
)
