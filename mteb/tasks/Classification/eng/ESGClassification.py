from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class ESGExtractionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ESGExtractionClassification",
        description="https://www.kaggle.com/datasets/tunguz/environment-social-and-governance-data",
        reference="https://huggingface.co/yiyanghkust/finbert-esg",
        dataset={
            "path": "yixuantt/esg",
            "revision": "11aa467ce80a54b771104911afe4daee3bcab73b",
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
        avg_character_length={"test": 171},
    )
