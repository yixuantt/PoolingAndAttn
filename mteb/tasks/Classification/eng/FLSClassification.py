from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class FLSExtractionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FLSExtractionClassification",
        description="https://www.kaggle.com/datasets/tunguz/environment-social-and-governance-data",
        reference="https://huggingface.co/yiyanghkust/finbert-fls",
        dataset={
            "path": "yixuantt/fls",
            "revision": "612aba72325bec24dc00399f075a8ac942723210",
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
        avg_character_length={"test": 188},
    )
