from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class FPBExtractionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FPBExtractionClassification",
        description="financial_phrasebank",
        reference="https://huggingface.co/datasets/financial_phrasebank",
        dataset={
            "path": "yixuantt/fpb",
            "revision": "616a340dae33f20c5229868a3286a5a45288335c",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 453},
        avg_character_length={"test": 119},
    )
