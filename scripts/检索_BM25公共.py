# -*- coding: utf-8 -*-
"""BM25 分词的公共约定。

建库与查询【必须】用同一套分词，否则同一个词在两侧被切成不同 token，命中率会
莫名其妙地掉。所以把分词器参数集中在这里，构建脚本和检索器都从这里取，改这里
就意味着要重建索引。

只依赖 bm25s + PyStemmer，不碰 torch/chroma，方便被轻量导入。

在链路里的位置：`检索_构建BM25索引.py`（建）与 `检索_多路检索.py`（查）共同 import 它，
是「两侧分词必须一致」这条约束的唯一落点。不单独运行。
"""
import bm25s
import Stemmer

# ---- 语料是纯英文（中文查询在查询理解层已译成英文），所以只需英文分词 ----
STOPWORDS = "en"                 # bm25s 内置英文停用词表
STEMMER_LANG = "english"         # snowball 词干化：diabetes/diabetic → diabet
BM25_METHOD = "lucene"           # 打分口径与 Elasticsearch 默认一致（Lucene BM25）
BM25_K1 = 1.5
BM25_B = 0.75

BM25_TOKENIZER_META = {
    "lib": "bm25s",
    "stopwords": STOPWORDS,
    "stemmer": STEMMER_LANG,
    "method": BM25_METHOD,
    "k1": BM25_K1,
    "b": BM25_B,
    "pipeline": "lowercase + r'\\w\\w+' 正则切词 + 英文停用词 + snowball 词干化",
}

_stemmer = None


def make_stemmer():
    """进程内单例的 snowball 词干器。"""
    global _stemmer
    if _stemmer is None:
        _stemmer = Stemmer.Stemmer(STEMMER_LANG)
    return _stemmer


def bm25_tokenize(texts, show_progress=False):
    """把文本或查询切成 bm25s 的 token 结构。

    建库时传 list[str]；查询时传 str 或 list[str] 均可（bm25s 会各自建小词表，
    retrieve 时按词面重映射到索引词表，因此两侧词表不同也没关系，只要分词规则一致）。
    """
    return bm25s.tokenize(
        texts, stopwords=STOPWORDS, stemmer=make_stemmer(),
        show_progress=show_progress,
    )
