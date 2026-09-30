"""Canonical Qwen3 non-thinking prompt rendering for single-teacher OPD."""

from slime.rollout.data_source import RolloutDataSource
from slime.rollout.multi_teacher_on_policy_distillation import parse_teacher_urls
from slime.rollout.paired_mode_on_policy_distillation import NON_THINKING, build_prompt_variant
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample


class CanonicalNonThinkingRolloutDataSource(RolloutDataSource):
    """Render student and Instruct-teacher prompts independently and align responses."""

    def __init__(self, args):
        if args.n_samples_per_prompt != 1:
            raise ValueError("Canonical single-teacher OPD requires --n-samples-per-prompt 1.")
        if args.apply_chat_template:
            raise ValueError("CanonicalNonThinkingRolloutDataSource renders chat templates itself.")
        super().__init__(args)
        self.teacher = args.paired_mopd_non_thinking_teacher
        teacher_paths = parse_teacher_urls(args.opd_teacher_tokenizers)
        if self.teacher not in teacher_paths:
            raise ValueError(f"Missing --opd-teacher-tokenizer for {self.teacher!r}.")
        self.student_tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.teacher_tokenizer = load_tokenizer(teacher_paths[self.teacher], trust_remote_code=True)
        self.forbidden_think_ids = {
            self.student_tokenizer.convert_tokens_to_ids("<think>"),
            self.student_tokenizer.convert_tokens_to_ids("</think>"),
        }

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = super().get_samples(num_samples)
        for group in groups:
            if len(group) != 1:
                raise AssertionError("Canonical single-teacher OPD group cardinality changed.")
            sample = group[0]
            variant = build_prompt_variant(
                prompt=sample.prompt,
                mode=NON_THINKING,
                teacher=self.teacher,
                student_tokenizer=self.student_tokenizer,
                teacher_tokenizer=self.teacher_tokenizer,
            )
            if len(variant.student_prompt_ids) > self.args.rollout_max_prompt_len:
                raise ValueError(
                    f"Preprocessed prompt exceeds {self.args.rollout_max_prompt_len} tokens in non-thinking mode."
                )
            if any(token_id in self.forbidden_think_ids for token_id in variant.teacher_prompt_ids):
                raise ValueError("Instruct teacher prompt unexpectedly contains a reserved think token.")
            sample.prompt = variant.student_prompt
            sample.metadata = dict(sample.metadata or {})
            sample.metadata.update(
                {
                    "teacher": self.teacher,
                    "mopd_mode": NON_THINKING,
                    "mopd_student_prompt_ids": list(variant.student_prompt_ids),
                    "mopd_teacher_prompt_ids": list(variant.teacher_prompt_ids),
                    "mopd_forbidden_response_token_ids": sorted(self.forbidden_think_ids),
                }
            )
        return groups
