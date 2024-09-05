from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class FinSentExtractionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinSentExtractionClassification",
        description="financial_sentiment",
        reference="https://cbsa.hkust.edu.hk/FinSent/",
        dataset={
            "path": "yixuantt/finsent",
            "revision": "51b57e355a0ede8b888250a5fdf95472c2e5ae2b",
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
        n_samples={"test": 1000},
        avg_character_length={"test": 139},
    )
