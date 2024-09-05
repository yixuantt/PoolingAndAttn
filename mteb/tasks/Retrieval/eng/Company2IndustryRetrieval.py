from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class Company2Industry(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Company2Industry",
        description="Financial Opinion Mining and Question Answering",
        reference="https://huggingface.co/datasets/yixuantt/company2industry",
        dataset={
            "path": "yixuantt/company2industry",
            "revision": "1cf8b07afe4c06199964f903f182473436e25774",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
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
        n_samples=None,
        avg_character_length=None,
    )
