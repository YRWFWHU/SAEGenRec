"""SFT training orchestrator — wraps trl SFTTrainer with SID token injection and LoRA."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from .callbacks import RecMetricsCallback
from .config import SFTTrainingConfig
from .dataset import add_sid_tokens_to_tokenizer, load_sft_dataset
from .evaluator import RecEvalResult, RecEvaluator


class SFTRecTrainer:
    """High-level orchestrator for SFT training on recommendation tasks.

    Handles model loading, SID token injection, LoRA application, dataset
    preparation, and training via trl's ``SFTTrainer``.
    """

    def __init__(self, config: SFTTrainingConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.sid_map = None

    def setup(self) -> None:
        """Load model, tokenizer, inject SID tokens, apply LoRA."""
        set_seed(self.seed)
        cfg = self.config
        self._validate_config()

        logger.info("Loading tokenizer: {}", cfg.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._ensure_generation_markers()

        if cfg.sid_map_path:
            logger.info("Loading SID map from {}", cfg.sid_map_path)
            self.sid_map = load_from_disk(cfg.sid_map_path)
            add_sid_tokens_to_tokenizer(self.tokenizer, self.sid_map)

        logger.info("Loading model: {}", cfg.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=True,
            dtype="auto",
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        if cfg.lora.enabled:
            modules_to_save = list(cfg.lora.modules_to_save) if cfg.lora.modules_to_save else []

            if (
                getattr(self.model.config, "tie_word_embeddings", False)
                and "embed_tokens" in modules_to_save
                and "lm_head" in modules_to_save
            ):
                modules_to_save.remove("lm_head")
                logger.info(
                    "Removed 'lm_head' from modules_to_save because "
                    "tie_word_embeddings=True (lm_head shares weights with embed_tokens)"
                )

            logger.info("Applying LoRA (r={}, alpha={})", cfg.lora.r, cfg.lora.alpha)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=cfg.lora.target_modules,
                modules_to_save=modules_to_save or None,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def train(self, extra_callbacks: list | None = None) -> None:
        """Run the SFT training loop."""
        cfg = self.config

        logger.info("Loading training data from {}", cfg.sft_data_dir)
        train_ds = load_sft_dataset(cfg.sft_data_dir, tasks=cfg.tasks, split="train")
        eval_ds = load_sft_dataset(cfg.sft_data_dir, tasks=cfg.tasks, split="valid")
        self._eval_ds = eval_ds

        sft_config = SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_length=cfg.max_seq_length,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            logging_steps=max(1, cfg.eval_steps // 5),
            report_to=cfg.report_to,
            seed=self.seed,
            assistant_only_loss=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
        )

        callbacks = list(extra_callbacks or [])

        if self.sid_map is not None:
            rec_evaluator = RecEvaluator(
                tokenizer=self.tokenizer,
                sid_map=self.sid_map,
                eval_top_k=cfg.eval_top_k,
                max_new_tokens=cfg.max_new_tokens,
            )
            self._rec_callback = RecMetricsCallback(
                evaluator=rec_evaluator,
                eval_dataset=eval_ds,
                rec_eval_steps=cfg.rec_eval_steps,
                eval_batch_size=cfg.per_device_eval_batch_size,
            )
            callbacks.append(self._rec_callback)
            logger.info("RecMetricsCallback enabled (every {} steps)", cfg.rec_eval_steps)

        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )

        logger.info("Starting SFT training...")
        self.trainer.train()
        logger.info("Training complete.")

        self.trainer.save_model(str(Path(cfg.output_dir) / "final_model"))
        self.tokenizer.save_pretrained(str(Path(cfg.output_dir) / "final_model"))
        logger.info("Saved final model to {}", Path(cfg.output_dir) / "final_model")

        if cfg.do_test:
            logger.info("Running post-training test evaluation...")
            self.test()

    # ChatML template with {% generation %} markers required by
    # trl's assistant_only_loss=True (v0.29.0+).
    _CHATML_TEMPLATE_WITH_GENERATION = (
        "{%- if messages[0]['role'] == 'system' -%}"
        "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}"
        "{%- else -%}"
        "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' -}}"
        "{%- endif -%}"
        "{%- for message in messages -%}"
        "{%- if message['role'] == 'system' and loop.first -%}"
        "{%- elif message['role'] == 'assistant' -%}"
        "{% generation %}"
        "{{- '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' -}}"
        "{% endgeneration %}"
        "{%- else -%}"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{- '<|im_start|>assistant\\n' -}}"
        "{%- endif -%}"
    )

    def _ensure_generation_markers(self) -> None:
        """Patch chat template to include {% generation %} markers if missing.

        trl v0.29.0+ requires these markers for ``assistant_only_loss=True``
        to know which tokens belong to the assistant response.
        """
        tpl = self.tokenizer.chat_template or ""
        if "{% generation %}" in tpl:
            return

        if "<|im_start|>" in tpl:
            logger.info(
                "Patching ChatML chat template with {{% generation %}} markers "
                "for assistant_only_loss support"
            )
            self.tokenizer.chat_template = self._CHATML_TEMPLATE_WITH_GENERATION
        else:
            logger.warning(
                "Chat template does not contain {{% generation %}} markers and "
                "is not ChatML format — assistant_only_loss may fail. "
                "Consider providing a custom chat template."
            )

    def _validate_config(self) -> None:
        """Validate config paths and parameters before setup."""
        cfg = self.config
        if cfg.sft_data_dir:
            sft_dir = Path(cfg.sft_data_dir)
            if not sft_dir.exists():
                raise FileNotFoundError(
                    f"SFT data directory not found at {sft_dir}. "
                    "Run 'python -m saegenrec.dataset build-sft <config> --splits train,valid,test' first."
                )
            missing = [s for s in ("train", "valid", "test") if not (sft_dir / s).exists()]
            if missing:
                raise FileNotFoundError(
                    f"SFT data splits {missing} not found under {sft_dir}. "
                    "Run 'python -m saegenrec.dataset build-sft <config> --splits train,valid,test' first."
                )
        if cfg.sid_map_path and not Path(cfg.sid_map_path).exists():
            raise FileNotFoundError(
                f"SID map not found at {cfg.sid_map_path}. "
                "Run 'python -m saegenrec.dataset tokenize' first."
            )

    def evaluate(self, split: str = "valid") -> dict:
        """Run loss evaluation on the given split."""
        cfg = self.config
        ds = load_sft_dataset(cfg.sft_data_dir, tasks=cfg.tasks, split=split)

        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call train() or setup() first.")

        metrics = self.trainer.evaluate(eval_dataset=ds)
        logger.info("Evaluation metrics ({}): {}", split, metrics)
        return metrics

    def test(self, checkpoint_path: str | None = None) -> RecEvalResult:
        """Run test set evaluation with constrained decoding.

        Args:
            checkpoint_path: If provided, loads model weights from this path.
                If ``None`` and a best checkpoint was tracked during training,
                uses that; otherwise uses the current model state.

        Returns:
            RecEvalResult with final test metrics.
        """
        cfg = self.config

        if checkpoint_path is None and hasattr(self, "_rec_callback"):
            checkpoint_path = self._rec_callback.best_checkpoint_path
            if checkpoint_path:
                logger.info("Using best checkpoint from training: {}", checkpoint_path)

        if checkpoint_path:
            logger.info("Loading checkpoint for testing: {}", checkpoint_path)
            from peft import PeftModel

            if cfg.lora.enabled and hasattr(self.model, "peft_config"):
                base_model = self.model.get_base_model()
                self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path, trust_remote_code=True, torch_dtype="auto"
                )
                self.model.resize_token_embeddings(len(self.tokenizer))

        test_ds = load_sft_dataset(cfg.sft_data_dir, tasks=["seqrec"], split="test")

        logits_processor = None
        num_beams = 1
        if cfg.constrained_decoding and self.sid_map is not None:
            from transformers import LogitsProcessorList

            from saegenrec.modeling.decoding.constrained import SIDConstrainedLogitsProcessor
            from saegenrec.modeling.decoding.trie import SIDTrie

            logger.info("Building SID trie for constrained decoding...")
            trie = SIDTrie.from_sid_map(self.sid_map, self.tokenizer)

            sid_begin_id = self.tokenizer.convert_tokens_to_ids("<|sid_begin|>")
            sid_end_id = self.tokenizer.convert_tokens_to_ids("<|sid_end|>")

            processor = SIDConstrainedLogitsProcessor(
                trie=trie,
                sid_begin_token_id=sid_begin_id,
                sid_end_token_id=sid_end_id,
            )
            logits_processor = LogitsProcessorList([processor])
            num_beams = 4
            logger.info("Constrained decoding enabled (beam_size={})", num_beams)

        evaluator = RecEvaluator(
            tokenizer=self.tokenizer,
            sid_map=self.sid_map,
            eval_top_k=cfg.eval_top_k,
            max_new_tokens=cfg.max_new_tokens,
        )

        result = evaluator.evaluate(
            model=self.model,
            eval_dataset=test_ds,
            step=-1,
            batch_size=cfg.per_device_eval_batch_size,
            logits_processor=logits_processor,
            num_beams=num_beams,
        )

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "step": result.step,
                    "num_samples": result.num_samples,
                    "num_valid_predictions": result.num_valid_predictions,
                    "metrics": result.metrics,
                    "constrained_decoding": cfg.constrained_decoding,
                    "checkpoint": checkpoint_path,
                },
                f,
                indent=2,
            )
        logger.info("Test results saved to {}", results_path)
        logger.info(
            "Test metrics: {}",
            {k: f"{v:.4f}" for k, v in result.metrics.items()},
        )

        return result
