TASK_LIST_CLASSIFICATION = [
    "AmazonPolarityClassification",
    "ToxicConversationsClassification",
    "ImdbClassification",
    "Banking77Classification",
    "EmotionClassification",
    "TweetSentimentExtractionClassification",
]
TASK_LIST_SUM = [
    "SummEval"
]
TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P.v2",
    "BiorxivClusteringS2S.v2",
    "MedrxivClusteringP2P.v2",
    "MedrxivClusteringS2S.v2",
    "RedditClustering.v2",
    "RedditClusteringP2P.v2",
    "StackExchangeClustering.v2",
    "StackExchangeClusteringP2P.v2",
    "TwentyNewsgroupsClustering.v2"
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "TwitterURLCorpus",
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "NFCorpus",
    "FiQA2018",
    "SciFact",
    "Touche2020",
    "QuoraRetrieval",
    "TRECCOVID",
    "NQ",
    "SCIDOCS",
    "ClimateFEVER",
    "DBPedia",
    "MSMARCO",
    "HotpotQA",
    "FEVER",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
]

STS_TASK_LIST = (
    TASK_LIST_STS
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_RERANKING
    + TASK_LIST_SUM
    + TASK_LIST_CLASSIFICATION
    + TASK_LIST_PAIR_CLASSIFICATION
)