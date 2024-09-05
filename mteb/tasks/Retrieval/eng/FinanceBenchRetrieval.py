from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FinanceBench(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinanceBench",
        description="Open book financial question answering (QA)",
        reference="https://huggingface.co/datasets/PatronusAI/financebench",
        dataset={
            "path": "yixuantt/financebench",
            "revision": "c8f475fc7eb8d129a422105641153e031cf722db",
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
