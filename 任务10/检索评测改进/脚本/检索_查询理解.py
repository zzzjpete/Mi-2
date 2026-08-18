# -*- coding: utf-8 -*-
"""
第五阶段 · 检索系统（一）· 查询理解与增强

输入：用户自然语言查询（中/英，含缩写、俗称、商品名、专业术语）
输出：EnhancedQuery —— 结构化的增强查询信息，供下游检索层直接消费
        · vector_queries  已加 BGE 指令前缀的稠密查询（主查询 + 消歧变体）
        · keyword_query   关键词/BM25 查询（组内 OR、组间 AND）
        · filters         Chroma where 子句（pub_year / journal / section）
        · post_filters    无法用 where 表达、需检索后再过滤的条件
        · entities/expansions  可解释性信息（识别到什么、扩展了什么、为什么）

设计要点（几处与「直觉做法」相反，理由写在下面，也写进了交付报告）：

  1. 同义词【不】拼进同一个向量查询。
     稠密检索里把 "MI (myocardial infarction, heart attack)" 塞成一句，会把查询向量
     拉向几个词的质心，反而稀释主题。正确做法是多查询（multi-query）：每个歧义/缩写
     生成一条独立查询，检索层各查一次再用 RRF 融合。所以 vector_queries 是【列表】。
     反过来，关键词/BM25 天生适合 OR 扩展，所以关键词侧才做同义词平铺。
     （本模块同时保留 vector_query_expanded 单查询版本，供 A/B 实测对照，见验证脚本。）

  2. 过滤短语要从查询里【剥掉】再送去做向量。
     "metformin cardiovascular outcomes since 2020" 里的 "since 2020" 已经变成
     where 子句了，留在文本里只会污染语义向量。

  3. section 过滤按语料实测【自适应】选实现方式。
     库里 section 原始取值有 39 万种写法；results/discussion 等规范类只需 ≤47 种写法
     即可覆盖 99%，可以直接下推成 Chroma $in；但 methods 需要 7726 种写法，$in 不现实，
     改为检索后用同一套归一化函数做后置过滤。阈值见 SECTION_IN_LIMIT。

  4. 中文查询必须先转英文。
     索引用的是 bge-base-en-v1.5（纯英文模型）+ 英文 PubMed 语料，中文查询直接向量化
     命中的是噪声。需求方给的示例查询「二甲双胍对心血管疾病有何影响？」正是中文，
     所以这里内置了中→英处理：词典直译（离线、零依赖）或本地 qwen3:8b 整句翻译。

依赖的外部产物（都由本阶段脚本生成，缺失时自动降级为纯静态词典）：
  data/dict/mesh_synonyms.json   ← 检索_构建同义词词典.py（MeSH 2026，2.7 万主题词）
  data/dict/corpus_meta.json     ← 检索_扫描元数据分布.py（语料年份/章节/期刊实测分布）

用法：
  # 单条查询，打印可读的增强结果
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_查询理解.py --query "Does MI risk increase with T2DM?"
  # 跑内置演示查询集
  E:\\rag\\conda\\envs\\medrag\\python.exe E:\\rag\\scripts\\检索_查询理解.py --demo

  # 作为库被下游检索层调用（中文文件名，按路径导入，与本项目既有脚本一致）
  import importlib.util
  spec = importlib.util.spec_from_file_location("qu", r"E:\\rag\\scripts\\检索_查询理解.py")
  qu = importlib.util.module_from_spec(spec); spec.loader.exec_module(qu)
  proc = qu.MedicalQueryProcessor()
  eq = proc.process_query("二甲双胍对心血管疾病有何影响？")
"""

# ── 项目根：同一份代码在 Windows 与 Mac 上都能跑（解析规则见 _medrag_root.py）──
import os
import sys
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _medrag_root import ROOT

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")     # 控制台 GBK，统一 UTF-8
except Exception:
    pass

MESH_DICT = os.path.join(ROOT, "data", "dict", "mesh_synonyms.json")
CORPUS_META = os.path.join(ROOT, "data", "dict", "corpus_meta.json")

# BGE 非对称检索的查询指令前缀。
#   OFFICIAL 是 BGE 官方写法，也是第四阶段建库/验证时用的那一条（保持一致）。
#   TASKBOOK 是任务书里写的 "question" 版本。二者实测差异见验证脚本 ③。
QUERY_INSTRUCTION_OFFICIAL = "Represent this sentence for searching relevant passages: "
QUERY_INSTRUCTION_TASKBOOK = "Represent this question for searching relevant passages: "

# section 规范类下的原始写法数超过这个阈值，就不下推 $in，改走后置过滤
SECTION_IN_LIMIT = 60


# ============================================================================
# 一、静态医学词典
#     任务书给的 MEDICAL_SYNONYMS 是扁平的 {词: [同义词...]}，这里保持同样的对外接口，
#     但内部按来源分成 4 张表，便于维护和标注置信度：
#       缩写表    —— 价值最高。MeSH 基本不收缩写（"MI" 不是 MeSH 入口词），只能手工维护。
#       俗称表    —— 患者用语 -> 医学术语。
#       商品名表  —— 商品名 -> 通用名。MeSH 主表只覆盖一部分（补充概念记录 supp2026.xml
#                    才收全，那是 3GB，本阶段未引入，作为后续扩展项写进报告）。
#       拼写表    —— 英式/美式拼写差异，PubMed 两种都有。
# ============================================================================

# 缩写 -> 全称。一个缩写有多个义项时全部列出，下游会为每个义项生成独立向量查询。
MEDICAL_ABBREVIATIONS = {
    # 循环
    "mi": ["myocardial infarction", "heart attack"],
    "chf": ["congestive heart failure"],
    "hf": ["heart failure"],
    "cad": ["coronary artery disease"],
    "cvd": ["cardiovascular disease"],
    "htn": ["hypertension"],
    "af": ["atrial fibrillation"],
    "afib": ["atrial fibrillation"],
    "acs": ["acute coronary syndrome"],
    "pci": ["percutaneous coronary intervention"],
    "cabg": ["coronary artery bypass grafting"],
    "lvef": ["left ventricular ejection fraction"],
    "dvt": ["deep vein thrombosis"],
    "vte": ["venous thromboembolism"],
    "pad": ["peripheral arterial disease"],
    "bp": ["blood pressure"],
    "ldl": ["low density lipoprotein cholesterol"],
    "hdl": ["high density lipoprotein cholesterol"],
    # 呼吸
    "copd": ["chronic obstructive pulmonary disease"],
    "ards": ["acute respiratory distress syndrome"],
    "osa": ["obstructive sleep apnea"],
    "ild": ["interstitial lung disease"],
    "ipf": ["idiopathic pulmonary fibrosis"],
    "tb": ["tuberculosis"],
    "cf": ["cystic fibrosis"],
    # 内分泌/代谢
    "dm": ["diabetes mellitus"],
    "t1dm": ["type 1 diabetes mellitus"],
    "t2dm": ["type 2 diabetes mellitus"],
    "t2d": ["type 2 diabetes mellitus"],
    "niddm": ["type 2 diabetes mellitus"],
    "dka": ["diabetic ketoacidosis"],
    "hba1c": ["glycated hemoglobin", "hemoglobin a1c"],
    "pcos": ["polycystic ovary syndrome"],
    "bmi": ["body mass index"],
    "nafld": ["non-alcoholic fatty liver disease"],
    "nash": ["non-alcoholic steatohepatitis"],
    # 肾
    "ckd": ["chronic kidney disease"],
    "aki": ["acute kidney injury"],
    "esrd": ["end stage renal disease"],
    "egfr_renal": ["estimated glomerular filtration rate"],
    "uti": ["urinary tract infection"],
    # 神经/精神
    "ad": ["alzheimer disease"],
    "pd": ["parkinson disease"],
    "ms": ["multiple sclerosis"],
    "tia": ["transient ischemic attack"],
    "cva": ["cerebrovascular accident", "stroke"],
    "tbi": ["traumatic brain injury"],
    "als": ["amyotrophic lateral sclerosis"],
    "ich": ["intracerebral hemorrhage"],
    "sah": ["subarachnoid hemorrhage"],
    "mdd": ["major depressive disorder"],
    "ptsd": ["post-traumatic stress disorder"],
    "adhd": ["attention deficit hyperactivity disorder"],
    "asd": ["autism spectrum disorder"],
    "ocd": ["obsessive compulsive disorder"],
    # 消化
    "ibd": ["inflammatory bowel disease"],
    "ibs": ["irritable bowel syndrome"],
    "uc": ["ulcerative colitis"],
    "gerd": ["gastroesophageal reflux disease"],
    "hcc": ["hepatocellular carcinoma"],
    # 肿瘤
    "nsclc": ["non-small cell lung carcinoma"],
    "sclc": ["small cell lung carcinoma"],
    "crc": ["colorectal neoplasms", "colorectal cancer"],
    "tnbc": ["triple negative breast neoplasms"],
    "aml": ["acute myeloid leukemia"],
    "cml": ["chronic myeloid leukemia"],
    "cll": ["chronic lymphocytic leukemia"],
    "rcc": ["renal cell carcinoma"],
    "gbm": ["glioblastoma"],
    "tme": ["tumor microenvironment"],
    "ici": ["immune checkpoint inhibitors"],
    "tki": ["tyrosine kinase inhibitors"],
    "car-t": ["chimeric antigen receptor t-cell therapy"],
    "os": ["overall survival"],
    "pfs": ["progression free survival"],
    "orr": ["objective response rate"],
    "tmb": ["tumor mutational burden"],
    # 感染/免疫
    "hiv": ["human immunodeficiency virus"],
    "aids": ["acquired immunodeficiency syndrome"],
    "hbv": ["hepatitis b virus"],
    "hcv": ["hepatitis c virus"],
    "hpv": ["papillomavirus"],
    "mrsa": ["methicillin resistant staphylococcus aureus"],
    "cmv": ["cytomegalovirus"],
    "ebv": ["epstein-barr virus"],
    "rsv": ["respiratory syncytial virus"],
    "sars-cov-2": ["severe acute respiratory syndrome coronavirus 2"],
    "covid": ["covid-19"],
    "amr": ["antimicrobial resistance"],
    # 风湿
    "ra": ["rheumatoid arthritis"],
    "sle": ["lupus erythematosus systemic"],
    "oa": ["osteoarthritis"],
    # 研究设计/统计（对科研语料很有用）
    "rct": ["randomized controlled trial"],
    "ci": ["confidence interval"],
    "rr": ["relative risk", "risk ratio"],
    "itt": ["intention to treat analysis"],
    "qol": ["quality of life"],
    "ae": ["adverse event"],
    "nnt": ["number needed to treat"],
    # 技术/检测
    "gwas": ["genome wide association study"],
    "snp": ["single nucleotide polymorphism"],
    "ngs": ["next generation sequencing", "high throughput nucleotide sequencing"],
    "wgs": ["whole genome sequencing"],
    "wes": ["whole exome sequencing"],
    "scrna-seq": ["single cell rna sequencing"],
    "qpcr": ["real time polymerase chain reaction"],
    "pcr": ["polymerase chain reaction"],
    "elisa": ["enzyme-linked immunosorbent assay"],
    "ihc": ["immunohistochemistry"],
    "mri": ["magnetic resonance imaging"],
    "ct": ["tomography x-ray computed", "computed tomography"],
    "pet": ["positron emission tomography"],
    "ecg": ["electrocardiography"],
    "ekg": ["electrocardiography"],
    "eeg": ["electroencephalography"],
    "ros": ["reactive oxygen species"],
    "emt": ["epithelial mesenchymal transition"],
}

# 俗称/患者用语 -> 医学术语
MEDICAL_LAY_TERMS = {
    "heart attack": ["myocardial infarction"],
    "stroke": ["cerebrovascular accident", "brain infarction"],
    "high blood pressure": ["hypertension"],
    "low blood pressure": ["hypotension"],
    "high cholesterol": ["hypercholesterolemia"],
    "blood sugar": ["blood glucose"],
    "sugar diabetes": ["diabetes mellitus"],
    "kidney failure": ["renal insufficiency"],
    "blood clot": ["thrombosis"],
    "blood thinner": ["anticoagulants"],
    "water pill": ["diuretics"],
    "painkiller": ["analgesics"],
    "pain killer": ["analgesics"],
    "swelling": ["edema"],
    "itching": ["pruritus"],
    "shortness of breath": ["dyspnea"],
    "heartburn": ["heartburn", "gastroesophageal reflux"],
    "fainting": ["syncope"],
    "bruising": ["contusions"],
    "hives": ["urticaria"],
    "kidney stones": ["kidney calculi", "nephrolithiasis"],
    "gallstones": ["cholelithiasis"],
    "hardening of the arteries": ["arteriosclerosis"],
    "irregular heartbeat": ["arrhythmias cardiac"],
    "flu": ["influenza"],
    "lung cancer": ["lung neoplasms"],
    "breast cancer": ["breast neoplasms"],
    "liver cancer": ["liver neoplasms"],
    "stomach cancer": ["stomach neoplasms"],
    "bowel cancer": ["colorectal neoplasms"],
}

# 商品名 -> 通用名
DRUG_BRAND_TO_GENERIC = {
    "tylenol": ["acetaminophen", "paracetamol"],
    "advil": ["ibuprofen"],
    "motrin": ["ibuprofen"],
    "aleve": ["naproxen"],
    "glucophage": ["metformin"],
    "lipitor": ["atorvastatin"],
    "crestor": ["rosuvastatin"],
    "zocor": ["simvastatin"],
    "coumadin": ["warfarin"],
    "plavix": ["clopidogrel"],
    "eliquis": ["apixaban"],
    "xarelto": ["rivaroxaban"],
    "prilosec": ["omeprazole"],
    "nexium": ["esomeprazole"],
    "zoloft": ["sertraline"],
    "prozac": ["fluoxetine"],
    "lasix": ["furosemide"],
    "norvasc": ["amlodipine"],
    "synthroid": ["levothyroxine"],
    "ozempic": ["semaglutide"],
    "wegovy": ["semaglutide"],
    "jardiance": ["empagliflozin"],
    "farxiga": ["dapagliflozin"],
    "januvia": ["sitagliptin"],
    "humira": ["adalimumab"],
    "keytruda": ["pembrolizumab"],
    "opdivo": ["nivolumab"],
    "herceptin": ["trastuzumab"],
    "avastin": ["bevacizumab"],
    "rituxan": ["rituximab"],
    "gleevec": ["imatinib"],
    "ventolin": ["albuterol", "salbutamol"],
}

# 英式 <-> 美式拼写（PubMed 两种写法都大量存在）
SPELLING_VARIANTS = {
    "tumour": ["tumor"], "tumours": ["tumors"],
    "haemorrhage": ["hemorrhage"], "haemorrhagic": ["hemorrhagic"],
    "anaemia": ["anemia"], "anaemic": ["anemic"],
    "oesophageal": ["esophageal"], "oesophagus": ["esophagus"],
    "paediatric": ["pediatric"], "paediatrics": ["pediatrics"],
    "leukaemia": ["leukemia"], "oedema": ["edema"],
    "diarrhoea": ["diarrhea"], "foetal": ["fetal"],
    "caesarean": ["cesarean"], "ischaemic": ["ischemic"],
    "ischaemia": ["ischemia"], "aetiology": ["etiology"],
    "anaesthesia": ["anesthesia"], "sulphate": ["sulfate"],
    "hospitalisation": ["hospitalization"], "randomised": ["randomized"],
    "characterisation": ["characterization"], "utilisation": ["utilization"],
}

# 中文 -> 英文（语料是纯英文，中文查询必须先落到英文术语上）
CN_EN_MEDICAL = {
    "二甲双胍": "metformin", "阿司匹林": "aspirin", "华法林": "warfarin",
    "胰岛素": "insulin", "他汀": "statins", "阿托伐他汀": "atorvastatin",
    "抗生素": "antibiotics", "疫苗": "vaccine", "免疫治疗": "immunotherapy",
    "化疗": "chemotherapy", "放疗": "radiotherapy", "靶向治疗": "targeted therapy",
    "心血管疾病": "cardiovascular disease", "心肌梗死": "myocardial infarction",
    "心力衰竭": "heart failure", "心衰": "heart failure", "高血压": "hypertension",
    "冠心病": "coronary artery disease", "动脉粥样硬化": "atherosclerosis",
    "中风": "stroke", "卒中": "stroke", "房颤": "atrial fibrillation",
    "糖尿病": "diabetes mellitus", "2型糖尿病": "type 2 diabetes mellitus",
    "二型糖尿病": "type 2 diabetes mellitus", "1型糖尿病": "type 1 diabetes mellitus",
    "肥胖": "obesity", "血脂": "lipids", "胆固醇": "cholesterol",
    "慢性肾病": "chronic kidney disease", "肾功能": "renal function",
    "肝硬化": "liver cirrhosis", "脂肪肝": "fatty liver",
    "哮喘": "asthma", "慢阻肺": "chronic obstructive pulmonary disease",
    "肺炎": "pneumonia", "结核": "tuberculosis",
    "癌症": "cancer", "肿瘤": "neoplasms", "肺癌": "lung neoplasms",
    "乳腺癌": "breast neoplasms", "结直肠癌": "colorectal neoplasms",
    "肝癌": "liver neoplasms", "胃癌": "stomach neoplasms",
    "肿瘤微环境": "tumor microenvironment", "转移": "neoplasm metastasis",
    "阿尔茨海默病": "alzheimer disease", "老年痴呆": "alzheimer disease",
    "帕金森病": "parkinson disease", "抑郁症": "depression", "焦虑": "anxiety",
    "炎症": "inflammation", "免疫": "immunity", "自身免疫": "autoimmunity",
    "类风湿关节炎": "rheumatoid arthritis", "骨质疏松": "osteoporosis",
    "新冠": "covid-19", "新冠肺炎": "covid-19", "流感": "influenza",
    "艾滋病": "acquired immunodeficiency syndrome", "乙肝": "hepatitis b",
    "肠道菌群": "gastrointestinal microbiome", "微生物组": "microbiome",
    "干细胞": "stem cells", "基因编辑": "gene editing", "基因治疗": "genetic therapy",
    "测序": "sequencing", "生物标志物": "biomarkers", "蛋白": "proteins",
    "临床试验": "clinical trial", "随机对照试验": "randomized controlled trial",
    "meta分析": "meta-analysis", "荟萃分析": "meta-analysis", "队列研究": "cohort studies",
    "机制": "mechanism", "疗效": "efficacy", "有效性": "effectiveness",
    "安全性": "safety", "副作用": "adverse effects", "不良反应": "adverse effects",
    "预后": "prognosis", "诊断": "diagnosis", "治疗": "treatment",
    "预防": "prevention", "risk因素": "risk factors", "危险因素": "risk factors",
    "发病率": "incidence", "死亡率": "mortality", "生存率": "survival rate",
    "儿童": "children", "老年人": "aged", "孕妇": "pregnant women",
    "影响": "effect", "作用": "effect", "关系": "association",
}

# 任务书要求的统一入口：把 4 张表合并成一张扁平词典
MEDICAL_SYNONYMS: Dict[str, List[str]] = {}
for _tbl in (MEDICAL_ABBREVIATIONS, MEDICAL_LAY_TERMS, DRUG_BRAND_TO_GENERIC, SPELLING_VARIANTS):
    for _k, _v in _tbl.items():
        MEDICAL_SYNONYMS.setdefault(_k, [])
        for _x in _v:
            if _x not in MEDICAL_SYNONYMS[_k]:
                MEDICAL_SYNONYMS[_k].append(_x)


# ============================================================================
# 二、医学实体正则
#     用单词边界 \b 保证完整匹配（避免 "aspirin" 里匹配出 "pirin" 这类问题）。
#     正则只覆盖高频硬核实体；泛化覆盖交给 MeSH 词典最长匹配（见 _match_gazetteer）。
# ============================================================================
MEDICAL_PATTERNS = {
    "drug": r"\b(aspirin|metformin|atorvastatin|simvastatin|rosuvastatin|warfarin|"
            r"insulin|clopidogrel|heparin|ibuprofen|acetaminophen|paracetamol|"
            r"amoxicillin|azithromycin|vancomycin|prednisone|dexamethasone|"
            r"semaglutide|empagliflozin|dapagliflozin|sitagliptin|"
            r"pembrolizumab|nivolumab|trastuzumab|bevacizumab|rituximab|imatinib|"
            r"tamoxifen|cisplatin|doxorubicin|methotrexate|statins?)\b",

    "disease": r"\b(diabetes|hypertension|hypotension|obesity|asthma|pneumonia|"
               r"tuberculosis|influenza|covid-19|sepsis|stroke|atherosclerosis|"
               r"myocardial infarction|heart failure|arrhythmia|cirrhosis|"
               r"alzheimer'?s? disease|parkinson'?s? disease|epilepsy|depression|"
               r"osteoporosis|arthritis|psoriasis|anemia|leukemia|lymphoma|"
               r"carcinoma|sarcoma|melanoma|glioblastoma|cancers?|tumou?rs?|neoplasms?)\b",

    # 基因/蛋白：命名家族用模式覆盖，避免逐个硬编码
    "gene_protein": r"\b(TP53|KRAS|NRAS|BRAF|EGFR|HER2|ERBB2|BRCA[12]|PTEN|MYC|ALK|"
                    r"PIK3CA|VEGFA?|APOE|MTHFR|ACE2|TNF-?α?|TGF-?β?|"
                    r"IL-?\d+[A-Za-z]?|CD\d+[A-Za-z]?|PD-?L?1|CTLA-?4|"
                    r"mTOR|AMPK|NF-?κ?B|STAT\d|JAK\d|CYP\d[A-Z]\d+|HLA-[A-Z]+\d*)\b",

    "procedure": r"\b(chemotherapy|radiotherapy|immunotherapy|surgery|transplantation|"
                 r"dialysis|angioplasty|biopsy|endoscopy|colonoscopy|vaccination|"
                 r"screening|CRISPR(?:-Cas9)?|gene editing|sequencing|"
                 r"magnetic resonance imaging|computed tomography)\b",

    # 研究设计：科研语料里非常有用，可直接驱动证据等级排序
    "study_design": r"\b(randomi[sz]ed controlled trials?|randomi[sz]ed trials?|"
                    r"meta-analys[ie]s|systematic reviews?|cohort stud(?:y|ies)|"
                    r"case-control stud(?:y|ies)|cross-sectional stud(?:y|ies)|"
                    r"clinical trials?|observational stud(?:y|ies)|"
                    r"double-blind|placebo-controlled|in vitro|in vivo)\b",

    # 剂量/测量值：带单位的数值
    "measurement": r"\b\d+(?:\.\d+)?\s?(?:mg|µg|ug|mcg|g|kg|ml|l|mmol/l|mg/dl|mmhg|"
                   r"iu|units?|%|fold|nm|µm|um)\b",
}
_COMPILED_PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in MEDICAL_PATTERNS.items()}


# ============================================================================
# 三、过滤条件抽取用的模式
# ============================================================================
_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
           "七": 7, "八": 8, "九": 9, "十": 10}
_EN_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# 英文停用词（关键词查询用；只删真停用词，不删 effect/risk 这类有信息量的词）
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "to", "from",
    "in", "on", "off", "over", "under", "again", "then", "once", "here", "there",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "would", "could", "may", "might", "must", "shall", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "does", "did", "am", "as", "s", "t",
}
# 「像医学词但没有区分度」的词。它们确实挂在 MeSH 树下（Risk、Therapeutics、
# Disease 都是正经主题词），但对检索没有指向性，扩展出来只会稀释查询：
#   risk → relative risk、treatment → therapeutics …
# 这些词不做实体、不做同义词扩展，但仍会作为普通词进入关键词查询。
_GENERIC_MEDICAL_NOISE = {
    "risk", "risks", "treatment", "treatments", "therapy", "therapies",
    "disease", "diseases", "disorder", "disorders", "syndrome", "syndromes",
    "patient", "patients", "study", "studies", "research", "trial", "trials",
    "effect", "effects", "outcome", "outcomes", "result", "results",
    "control", "controls", "analysis", "analyses", "response", "responses",
    "function", "functions", "activity", "activities", "method", "methods",
    "care", "health", "safety", "efficacy", "evidence", "review", "reviews",
    "level", "levels", "group", "groups", "factor", "factors", "role", "roles",
    "use", "uses", "usage", "management", "association", "associations",
    "prevention", "diagnosis", "prognosis", "mechanism", "mechanisms",
    "incidence", "prevalence", "mortality", "survival", "population",
    "adult", "adults", "child", "children", "elderly", "human", "humans",
    "cell", "cells", "tissue", "tissues", "protein", "proteins", "gene", "genes",
}

# 查询开头的客套/引导语，剥掉可减少稠密向量噪声
_FILLER_PREFIXES = [
    r"^(please\s+)?(can|could|would)\s+you\s+(please\s+)?(tell\s+me|explain|describe)\s*",
    r"^i\s+(want|would\s+like)\s+to\s+know\s+(about\s+)?",
    r"^(please\s+)?(tell\s+me|explain|describe)\s+(about\s+)?",
    r"^what\s+(do|does)\s+(the\s+)?(literature|studies|research|evidence)\s+say\s+(about\s+)?",
    r"^(请问|请|麻烦|我想知道|想了解一下|想问一下|帮我查一下|帮我找一下)\s*",
]


# ============================================================================
# 四、数据结构
# ============================================================================
@dataclass
class Entity:
    """识别到的医学实体。"""
    text: str                 # 原文中的字面
    start: int
    end: int
    etype: str                # drug / disease / procedure / gene_protein / study_design / ...
    source: str               # regex | static | mesh
    norm: str = ""            # 规范名（MeSH 首选词 / 静态词典首选展开）
    ui: str = ""              # MeSH DescriptorUI
    confidence: str = "high"  # high | medium
    ambiguous: bool = False   # 该缩写有多个义项

    def brief(self):
        amb = " ⚠歧义" if self.ambiguous else ""
        norm = f" → {self.norm}" if self.norm and self.norm != self.text.lower() else ""
        return f"{self.text}[{self.etype}/{self.source}]{norm}{amb}"


@dataclass
class EnhancedQuery:
    """查询理解与增强的完整输出，下游检索层直接消费这个对象。"""
    original: str
    cleaned: str
    language: str = "en"                       # en / zh / mixed
    translated_from: Optional[str] = None      # 中文原句（若发生过翻译）
    translate_method: Optional[str] = None     # dict | llm | None

    entities: List[Entity] = field(default_factory=list)
    expansions: Dict[str, List[str]] = field(default_factory=dict)

    filters: Dict = field(default_factory=dict)        # Chroma where 子句
    post_filters: Dict = field(default_factory=dict)   # 需检索后再过滤的条件
    filter_evidence: List[str] = field(default_factory=list)

    vector_queries: List[str] = field(default_factory=list)   # 已带指令前缀，[0] 为主查询
    vector_query_weights: List[float] = field(default_factory=list)  # 与 vector_queries 一一对应
    vector_query_expanded: str = ""            # 同义词平铺进单条查询（A/B 对照用，默认不用）
    core_text: str = ""                        # 剥掉过滤短语与客套语后的查询主体（未加前缀）

    keyword_groups: List[List[str]] = field(default_factory=list)  # 组内 OR，组间 AND
    keyword_query: str = ""

    notes: List[str] = field(default_factory=list)     # 处理过程中的提示/告警

    @property
    def vector_query(self) -> str:
        """单查询场景的便捷入口（= 主查询）。"""
        return self.vector_queries[0] if self.vector_queries else ""

    def to_dict(self):
        d = asdict(self)
        d["vector_query"] = self.vector_query
        return d

    def pretty(self) -> str:
        L = []
        L.append(f"原始查询   : {self.original}")
        if self.cleaned != self.original:
            L.append(f"清洗后     : {self.cleaned}")
        if self.translated_from:
            L.append(f"中译英     : {self.translated_from}  →  {self.core_text}  （{self.translate_method}）")
        L.append(f"语言       : {self.language}")
        L.append(f"检索主体   : {self.core_text}")
        L.append(f"医学实体   : {[e.brief() for e in self.entities] or '（未识别到）'}")
        if self.expansions:
            L.append("同义词扩展 :")
            for k, v in self.expansions.items():
                L.append(f"             {k} → {v}")
        else:
            L.append("同义词扩展 : （无）")
        L.append(f"过滤条件   : {json.dumps(self.filters, ensure_ascii=False) if self.filters else '（无）'}")
        if self.post_filters:
            L.append(f"后置过滤   : {json.dumps(self.post_filters, ensure_ascii=False)}")
        if self.filter_evidence:
            L.append(f"过滤依据   : {self.filter_evidence}")
        L.append(f"向量查询   : {len(self.vector_queries)} 条（权重供检索层做加权 RRF）")
        for i, q in enumerate(self.vector_queries):
            tag = "主" if i == 0 else f"变体{i}"
            w = self.vector_query_weights[i] if i < len(self.vector_query_weights) else 1.0
            L.append(f"             [{tag} w={w:g}] {q}")
        L.append(f"关键词查询 : {self.keyword_query}")
        if self.notes:
            L.append("提示       :")
            for n in self.notes:
                L.append(f"             · {n}")
        return "\n".join(L)


# ============================================================================
# 五、处理器
# ============================================================================
class MedicalQueryProcessor:
    """医学查询理解与增强。

    process_query() 是唯一对外入口，内部按任务书的六步走：
      基础清洗 → 语言处理 → 提取过滤条件 → 识别医学实体 → 同义词扩展 → 生成查询版本
    """

    def __init__(self,
                 mesh_dict_path: str = MESH_DICT,
                 corpus_meta_path: str = CORPUS_META,
                 use_mesh: bool = True,
                 query_instruction: str = QUERY_INSTRUCTION_OFFICIAL,
                 max_expansions_per_entity: int = 3,
                 max_total_expansions: int = 8,
                 max_vector_variants: int = 3,
                 primary_weight_when_expanded: float = 0.5,
                 recency_years: int = 5,
                 translate: str = "dict",          # off | dict | llm
                 ollama_model: str = "qwen3:8b",
                 ollama_url: str = "http://localhost:11434",
                 ollama_timeout: int = 90,
                 verbose: bool = False):
        self.instr = query_instruction
        self.max_exp_ent = max_expansions_per_entity
        self.max_exp_total = max_total_expansions
        self.max_variants = max_vector_variants
        self.primary_weight_when_expanded = primary_weight_when_expanded
        self.recency_years = recency_years
        self.translate_mode = translate
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_timeout = ollama_timeout
        self.verbose = verbose

        # --- MeSH 词典（缺失则降级为纯静态词典，不报错）---
        self.mesh_index, self.mesh_desc = {}, {}
        if use_mesh and os.path.exists(mesh_dict_path):
            with open(mesh_dict_path, encoding="utf-8") as f:
                d = json.load(f)
            self.mesh_index = d["index"]
            self.mesh_desc = d["descriptors"]
            if verbose:
                print(f"[词典] MeSH {len(self.mesh_desc):,} 主题词 / {len(self.mesh_index):,} 词面", flush=True)
        elif use_mesh:
            print(f"[词典] 警告：未找到 {mesh_dict_path}，降级为纯静态词典（覆盖会明显变窄）", flush=True)

        # --- 语料实测元数据（用于让过滤条件落到真实取值上）---
        self.corpus = None
        self.section_map = {}
        if os.path.exists(corpus_meta_path):
            with open(corpus_meta_path, encoding="utf-8") as f:
                self.corpus = json.load(f)
            self.section_map = self.corpus["section"]["canonical_to_raw"]

        # gazetteer：静态词典的键 + MeSH 全部词面，按词数分桶以便最长匹配
        self.gazetteer = set(self.mesh_index.keys()) | set(MEDICAL_SYNONYMS.keys())
        self.max_ngram = 6

    # ---------------- 步骤 1：基础清洗 ----------------
    def _clean(self, q: str):
        notes = []
        s = unicodedata.normalize("NFKC", q)        # 全角→半角、兼容字符归一
        s = s.replace(" ", " ")
        s = re.sub(r"\s+", " ", s).strip()
        s = s.strip("\"'“”‘’ 　")
        for pat in _FILLER_PREFIXES:                # 剥掉客套/引导语
            new = re.sub(pat, "", s, flags=re.IGNORECASE)
            if new != s:
                notes.append(f"剥离引导语：{s[:len(s)-len(new)]!r}")
                s = new.strip()
        return s, notes

    # ---------------- 步骤 2：语言判定与中译英 ----------------
    @staticmethod
    def _detect_lang(s: str) -> str:
        cjk = len(re.findall(r"[一-鿿]", s))
        latin = len(re.findall(r"[A-Za-z]", s))
        if cjk == 0:
            return "en"
        return "zh" if cjk >= latin else "mixed"

    def _translate_by_dict(self, s: str):
        """词典直译：按中文词长从长到短替换，剩余中文字符丢弃（它们多是虚词/助词）。"""
        hit = []
        out = s
        for cn in sorted(CN_EN_MEDICAL, key=len, reverse=True):
            if cn in out:
                out = out.replace(cn, " " + CN_EN_MEDICAL[cn] + " ")
                hit.append(cn)
        leftover = re.findall(r"[一-鿿]+", out)
        out = re.sub(r"[一-鿿]+", " ", out)
        out = re.sub(r"[？?。，,、；;！!]", " ", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out, hit, leftover

    def _translate_by_llm(self, s: str):
        """本地 qwen3:8b 整句翻译。Ollama 不可达时返回 None，由调用方降级。

        冷启动要把 5GB 权重装进显存，实测首次调用约 60s、之后每次约 5s。
        所以第一次超时不算失败——重试一次（此时模型多半已加载完），避免静默降级成词典直译。
        """
        import urllib.request
        prompt = ("Translate the following Chinese biomedical question into a concise English "
                  "search query. Use standard medical terminology. Output ONLY the English query, "
                  "no explanation, no quotes.\n\n" + s + " /no_think")
        body = json.dumps({"model": self.ollama_model, "prompt": prompt,
                           "stream": False, "options": {"temperature": 0}}).encode("utf-8")
        for attempt in (1, 2):
            try:
                req = urllib.request.Request(self.ollama_url + "/api/generate", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.ollama_timeout) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                out = resp.get("response", "")
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)   # qwen3 思考块
                out = out.strip().strip('"').strip()
                return out.split("\n")[0].strip() or None
            except Exception as e:
                is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                if attempt == 1 and is_timeout:
                    if self.verbose:
                        print(f"[翻译] 首次调用超时（模型冷启动中），重试一次 ...", flush=True)
                    continue
                if self.verbose:
                    print(f"[翻译] Ollama 不可用（{type(e).__name__}），降级词典直译", flush=True)
                return None

    # ---------------- 步骤 3：提取过滤条件 ----------------
    def _extract_filters(self, s: str):
        """从查询中抽出元数据过滤条件，并把命中的短语从文本里剥掉。

        返回 (filters, post_filters, evidence, residual_text, notes)
        """
        now_year = datetime.now().year
        conds, post, ev, notes = [], {}, [], []
        text = s
        spans = []

        def take(m, desc):
            spans.append((m.start(), m.end()))
            ev.append(f"{m.group(0).strip()!r} → {desc}")

        # --- 年份区间：between X and Y ---
        for m in re.finditer(r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})\b", text, re.I):
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = min(a, b), max(a, b)
            conds += [{"pub_year": {"$gte": lo}}, {"pub_year": {"$lte": hi}}]
            take(m, f"pub_year {lo}–{hi}")
        # --- 中文年份区间 ---
        for m in re.finditer(r"((?:19|20)\d{2})\s*[-–~至到]\s*((?:19|20)\d{2})\s*年?", text):
            a, b = int(m.group(1)), int(m.group(2))
            conds += [{"pub_year": {"$gte": min(a, b)}}, {"pub_year": {"$lte": max(a, b)}}]
            take(m, f"pub_year {min(a,b)}–{max(a,b)}")
        # --- 下界 ---
        for m in re.finditer(r"\b(?:since|after|from|later\s+than|newer\s+than)\s+((?:19|20)\d{2})\b", text, re.I):
            y = int(m.group(1))
            conds.append({"pub_year": {"$gte": y}})
            take(m, f"pub_year ≥ {y}")
        for m in re.finditer(r"((?:19|20)\d{2})\s*年?\s*(?:以来|之后|以后|后)", text):
            y = int(m.group(1))
            conds.append({"pub_year": {"$gte": y}})
            take(m, f"pub_year ≥ {y}")
        # --- 上界 ---
        for m in re.finditer(r"\b(?:before|prior\s+to|until|up\s+to|earlier\s+than)\s+((?:19|20)\d{2})\b", text, re.I):
            y = int(m.group(1))
            conds.append({"pub_year": {"$lt": y}})
            take(m, f"pub_year < {y}")
        for m in re.finditer(r"((?:19|20)\d{2})\s*年?\s*(?:以前|之前|前)", text):
            y = int(m.group(1))
            conds.append({"pub_year": {"$lt": y}})
            take(m, f"pub_year < {y}")
        # --- 近 N 年 ---
        for m in re.finditer(r"\b(?:in\s+the\s+)?(?:last|past|recent)\s+(\d+|" +
                             "|".join(_EN_NUM) + r")\s+years?\b", text, re.I):
            g = m.group(1).lower()
            n = int(g) if g.isdigit() else _EN_NUM[g]
            conds.append({"pub_year": {"$gte": now_year - n}})
            take(m, f"pub_year ≥ {now_year - n}（近 {n} 年）")
        for m in re.finditer(r"(?:近|最近|过去)\s*(\d+|[一两二三四五六七八九十])\s*年", text):
            g = m.group(1)
            n = int(g) if g.isdigit() else _CN_NUM.get(g, self.recency_years)
            conds.append({"pub_year": {"$gte": now_year - n}})
            take(m, f"pub_year ≥ {now_year - n}（近 {n} 年）")
        # --- 模糊近期（启发式，会在 notes 里声明）---
        if not conds:
            m = re.search(r"\b(recent|latest|newest|up-to-date|state[- ]of[- ]the[- ]art)\b", text, re.I) \
                or re.search(r"(最新|最近的研究|近期)", text)
            if m:
                y = now_year - self.recency_years
                conds.append({"pub_year": {"$gte": y}})
                take(m, f"pub_year ≥ {y}（模糊「近期」，按 {self.recency_years} 年启发式）")
                notes.append(f"「{m.group(0)}」是模糊时间词，按启发式取近 {self.recency_years} 年；"
                             f"如需关闭，构造时传 recency_years=0")

        # --- 章节 ---
        sec_pat = (r"\b(?:in|from|within)\s+the\s+"
                   r"(methods?|materials?\s+and\s+methods?|results?|discussions?|"
                   r"conclusions?|introductions?|abstracts?)\s*(?:section|part)?\b")
        m = re.search(sec_pat, text, re.I)
        if m is None:
            m = re.search(r"(?:只看|仅看|限定)\s*(方法|结果|讨论|结论|引言|摘要)\s*(?:部分|章节)?", text)
            cn2canon = {"方法": "methods", "结果": "results", "讨论": "discussion",
                        "结论": "conclusion", "引言": "introduction", "摘要": "abstract"}
            canon = cn2canon.get(m.group(1)) if m else None
        else:
            raw = m.group(1).lower()
            canon = ("methods" if "method" in raw or "material" in raw else
                     "results" if raw.startswith("result") else
                     "discussion" if raw.startswith("discussion") else
                     "conclusion" if raw.startswith("conclusion") else
                     "introduction" if raw.startswith("introduction") else
                     "abstract" if raw.startswith("abstract") else None)
        if m is not None and canon:
            variants = self.section_map.get(canon, [])
            if variants and len(variants) <= SECTION_IN_LIMIT:
                conds.append({"section": {"$in": variants}})
                take(m, f"section ∈ {canon}（下推 $in，{len(variants)} 种写法）")
            else:
                post["section_canon"] = canon
                if not self.section_map:
                    take(m, f"section = {canon}（无语料元数据，改后置过滤）")
                    notes.append(f"未加载 corpus_meta.json，不知道 section「{canon}」在库里有哪些写法，"
                                 f"已改为检索后按归一化规则过滤（跑 检索_扫描元数据分布.py 可启用 $in 下推）")
                else:
                    take(m, f"section = {canon}（写法 {len(variants)} 种，超出 $in 阈值，改后置过滤）")
                    notes.append(f"section「{canon}」在语料里有 {len(variants)} 种写法，"
                                 f"下推 $in 不现实，已改为检索后按同一套归一化规则过滤")

        # --- 期刊 ---
        m = re.search(r"\b(?:published\s+in|from\s+the\s+journal|in\s+the\s+journal)\s+"
                      r"([A-Z][A-Za-z&.\- ]{2,40}?)(?:\s+(?:since|after|before|between)\b|[,.?]|$)", text)
        if m:
            j = m.group(1).strip()
            conds.append({"journal": j})
            take(m, f"journal = {j!r}")
            notes.append(f"期刊名按字面精确匹配 {j!r}；库里期刊名是全称"
                         f"（如 'PLoS ONE'、'Sensors (Basel, Switzerland)'），不匹配会返回空")

        # 剥掉命中的过滤短语，避免污染稠密向量
        residual = text
        for a, b in sorted(spans, reverse=True):
            residual = residual[:a] + " " + residual[b:]
        residual = re.sub(r"\s+", " ", residual).strip(" ,.;:-")

        # 组装 Chroma where：单条件直接给，多条件用 $and
        if not conds:
            filters = {}
        elif len(conds) == 1:
            filters = conds[0]
        else:
            filters = {"$and": conds}

        # 年份是否落在语料覆盖范围外
        if self.corpus:
            lo_req = max([c["pub_year"]["$gte"] for c in conds
                          if "pub_year" in c and "$gte" in c["pub_year"]] or [None])
            ymax = self.corpus["pub_year"]["max"]
            hist = self.corpus["pub_year"]["histogram"]
            if lo_req is not None:
                n_hit = sum(v for k, v in hist.items() if int(k) >= lo_req)
                if n_hit == 0:
                    notes.append(f"⚠ 语料里没有 {lo_req} 年及以后的文献（最新 {ymax}），该过滤会返回空")
                elif n_hit < 20000:
                    notes.append(f"⚠ 满足 pub_year≥{lo_req} 的块只有 {n_hit:,} 条，召回可能不足")

        return filters, post, ev, residual, notes

    # ---------------- 步骤 4：识别医学实体 ----------------
    @staticmethod
    def _word_spans(text: str):
        return [(m.start(), m.end()) for m in
                re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-'’+/.]*", text)]

    def _match_gazetteer(self, text: str, taken):
        """词典最长匹配：从长 n-gram 到短，命中即占位，避免重叠。"""
        out = []
        spans = self._word_spans(text)
        n = len(spans)
        used = [False] * n
        for size in range(self.max_ngram, 0, -1):
            for i in range(0, n - size + 1):
                if any(used[i:i + size]):
                    continue
                a, b = spans[i][0], spans[i + size - 1][1]
                if any(not (b <= x or a >= y) for x, y in taken):   # 与正则结果重叠
                    continue
                surface = text[a:b]
                key = re.sub(r"\s+", " ", surface).strip().lower().rstrip(".")
                if key not in self.gazetteer:
                    continue
                # 单词命中的降噪
                if size == 1:
                    if key in _STOPWORDS or key in _GENERIC_MEDICAL_NOISE:
                        continue
                    in_static = key in MEDICAL_SYNONYMS
                    # 缩写表里的词允许短到 2 字符（"MI" 是任务书的示例，必须能识别）；
                    # 其余单词至少 3 字符
                    if len(key) < 3 and not (in_static and key in MEDICAL_ABBREVIATIONS):
                        continue
                    uis = self.mesh_index.get(key, [])
                    mtype = self.mesh_desc[uis[0]]["type"] if uis else None
                    # MeSH 单词命中只信疾病/药物/操作三类，且长度 ≥4（解剖/生物名单词误报率高）
                    if not in_static and not (mtype in ("disease", "drug", "procedure") and len(key) >= 4):
                        continue
                uis = self.mesh_index.get(key, [])
                if uis:
                    d = self.mesh_desc[uis[0]]
                    etype, norm, ui, src = d["type"], d["terms"][0], uis[0], "mesh"
                else:
                    etype, norm, ui, src = self._static_type(key), key, "", "static"
                amb = key in MEDICAL_ABBREVIATIONS and len(MEDICAL_ABBREVIATIONS[key]) > 1
                # 缩写用小写形式出现时置信度降一档（"mi" 也可能只是普通字母串）
                conf = "high"
                if key in MEDICAL_ABBREVIATIONS and len(key) <= 4 and surface != surface.upper():
                    conf = "medium"
                out.append(Entity(text=surface, start=a, end=b, etype=etype, source=src,
                                  norm=norm, ui=ui, confidence=conf, ambiguous=amb))
                for k in range(i, i + size):
                    used[k] = True
        return out

    @staticmethod
    def _static_type(key: str) -> str:
        if key in DRUG_BRAND_TO_GENERIC:
            return "drug"
        if key in MEDICAL_ABBREVIATIONS:
            return "abbreviation"
        if key in MEDICAL_LAY_TERMS:
            return "lay_term"
        if key in SPELLING_VARIANTS:
            return "spelling_variant"
        return "medical_term"

    def extract_entities(self, text: str) -> List[Entity]:
        """正则（任务书方案）+ 词典最长匹配（MeSH gazetteer）双路识别。

        两路【独立】取候选，最后按「跨度长者优先」统一裁决重叠。
        不能让某一路先占位：正则里的 `cancers?` 会先咬住 "cancer"，
        把词典里更具体的 "non-small cell lung cancer" 挡在门外——具体术语才是检索要的。
        跨度相同时优先正则（它带确定的类型标注）。
        """
        cands = []
        for etype, pat in _COMPILED_PATTERNS.items():
            for m in pat.finditer(text):
                surface = m.group(0)
                key = surface.lower().strip()
                uis = self.mesh_index.get(key, [])
                cands.append(Entity(text=surface, start=m.start(), end=m.end(),
                                    etype=etype, source="regex",
                                    norm=self.mesh_desc[uis[0]]["terms"][0] if uis else key,
                                    ui=uis[0] if uis else ""))
        cands += self._match_gazetteer(text, taken=[])

        # 长跨度优先；同长度时正则优先；再按出现位置稳定排序
        cands.sort(key=lambda e: (-(e.end - e.start), 0 if e.source == "regex" else 1, e.start))
        ents, taken = [], []
        for e in cands:
            if any(not (e.end <= x or e.start >= y) for x, y in taken):
                continue
            ents.append(e)
            taken.append((e.start, e.end))
        ents.sort(key=lambda e: e.start)
        return ents

    # ---------------- 步骤 5：同义词扩展 ----------------
    @staticmethod
    def _mesh_syn_ok(t: str) -> bool:
        """过滤 MeSH 里对检索没用的词面（化学全名、编号、超长串）。"""
        if len(t) > 45 or len(t) < 3:
            return False
        if re.search(r"\d{4,}", t):
            return False
        if t.count("-") >= 3 or t.count(",") >= 2:
            return False
        if re.match(r"^[a-z]{1,3}\s?\d", t):          # "n 4"、"ac 12" 之类编号
            return False
        return True

    def expand_synonyms(self, entities: List[Entity]) -> Dict[str, List[str]]:
        """按实体做同义词扩展。静态词典优先（缩写/俗称/商品名），MeSH 补覆盖。"""
        exp: Dict[str, List[str]] = {}
        budget = self.max_exp_total
        for e in entities:
            if budget <= 0:
                break
            key = e.text.lower().strip().rstrip(".")
            cand: List[str] = []
            # a) 静态词典
            for s in MEDICAL_SYNONYMS.get(key, []):
                if s not in cand:
                    cand.append(s)
            # b) MeSH 同主题词下的其余入口词
            uis = self.mesh_index.get(key, []) or ([e.ui] if e.ui else [])
            for ui in uis[:1]:
                d = self.mesh_desc.get(ui)
                if not d:
                    continue
                for t in d["terms"]:
                    if t != key and t not in cand and self._mesh_syn_ok(t):
                        cand.append(t)
            # c) 静态展开项本身在 MeSH 里的规范名（如 mi → myocardial infarction → 其入口词）
            for s in list(cand)[:1]:
                for ui in self.mesh_index.get(s, [])[:1]:
                    for t in self.mesh_desc[ui]["terms"]:
                        if t not in cand and t != key and self._mesh_syn_ok(t):
                            cand.append(t)
            cand = cand[:self.max_exp_ent]
            if cand:
                exp[e.text] = cand
                budget -= len(cand)
        return exp

    # ---------------- 步骤 6：生成不同的查询版本 ----------------
    def _build_vector_queries(self, core: str, entities: List[Entity],
                              expansions: Dict[str, List[str]]):
        """主查询 + 消歧变体，并给出各自的融合权重。

        变体【只】为缩写/歧义实体生成——把 "MI" 换成 "myocardial infarction" 是在补全
        语义，而不是稀释；给已经写全的术语再造变体没有收益，只会增加检索成本。

        权重的由来（实测驱动，见验证报告 ④）：
        查询 "Does MI risk increase in patients with CKD?" 在等权 RRF 下 term-hit@10
        只有 0.40，而单看展开后的变体能到 0.70——因为主查询里的 "MI"/"CKD" 没被模型
        理解，它的结果质量最差，等权融合等于让最差的一路拉低整体。
        因此：一旦主查询里的缩写已经在变体中被展开，就给主查询降权（默认 0.5），
        由检索层做加权 RRF：score += w / (k + rank)。
        """
        queries, weights = [core], [1.0]
        for e in entities:
            if len(queries) > self.max_variants:
                break
            key = e.text.lower().strip()
            if key not in MEDICAL_ABBREVIATIONS and e.etype not in ("abbreviation",):
                continue
            for full in MEDICAL_ABBREVIATIONS.get(key, [])[:2]:
                v = core[:e.start] + full + core[e.end:] if 0 <= e.start <= len(core) else None
                if v is None or full.lower() in core.lower():
                    # 位置对不上（core 被剥过）或全称已在句中：退化为整体替换
                    v = re.sub(rf"\b{re.escape(e.text)}\b", full, core, flags=re.IGNORECASE)
                v = re.sub(r"\s+", " ", v).strip()
                if v and v.lower() != core.lower() and v not in queries:
                    queries.append(v)
                    weights.append(1.0)
                if len(queries) > self.max_variants:
                    break
        if len(queries) > 1:
            weights[0] = self.primary_weight_when_expanded
        return [self.instr + q for q in queries], weights

    def _build_keyword_query(self, core: str, entities: List[Entity],
                             expansions: Dict[str, List[str]]):
        """关键词/BM25 查询：每个实体一组（组内 OR 同义词），其余实词各成一组，组间 AND。"""
        groups, consumed = [], set()
        for e in entities:
            terms = [e.text.lower()]
            if e.norm and e.norm.lower() not in terms:
                terms.append(e.norm.lower())
            for s in expansions.get(e.text, []):
                if s.lower() not in terms:
                    terms.append(s.lower())
            groups.append(terms)
            # 实体字面本身、以及它切出来的子词都算已消费，
            # 否则 "CRISPR-Cas9" 会在下面的散词扫描里作为整串再进一次
            consumed.add(e.text.lower())
            for w in re.findall(r"[a-z0-9]+", e.text.lower()):
                consumed.add(w)
        for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", core.lower()):
            if w in _STOPWORDS or w in consumed or len(w) < 3:
                continue
            if any(part in consumed for part in re.findall(r"[a-z0-9]+", w)):
                continue
            consumed.add(w)
            groups.append([w])
        parts = []
        for g in groups:
            g2 = [f'"{t}"' if " " in t else t for t in g]
            parts.append(f"({' OR '.join(g2)})" if len(g2) > 1 else g2[0])
        return groups, " AND ".join(parts)

    # ---------------- 主入口 ----------------
    def process_query(self, query: str) -> EnhancedQuery:
        """处理医学查询，返回增强的查询信息。"""
        eq = EnhancedQuery(original=query, cleaned=query)

        # 1) 基础清洗
        cleaned, notes = self._clean(query)
        eq.notes += notes
        eq.cleaned = cleaned
        eq.language = self._detect_lang(cleaned)

        # 2) 提取过滤条件（并把过滤短语从正文剥掉）
        #    必须在翻译【之前】做：中文时间词（「近五年」「2020年以来」）走的是中文正则，
        #    翻译一旦先跑，这些词会被当作未覆盖片段丢掉，过滤条件就永远抽不出来。
        filters, post, ev, core, fnotes = self._extract_filters(cleaned)
        eq.filters, eq.post_filters, eq.filter_evidence = filters, post, ev
        eq.notes += fnotes
        core = core or cleaned

        # 3) 中译英（语料与嵌入模型都是英文，中文查询必须先落到英文术语上）
        if eq.language in ("zh", "mixed") and self.translate_mode != "off":
            src = core
            out = None
            if self.translate_mode == "llm":
                out = self._translate_by_llm(core)
                if out:
                    eq.translate_method = "llm"
            if not out:
                out, hit, leftover = self._translate_by_dict(core)
                eq.translate_method = "dict"
                if leftover:
                    eq.notes.append(f"中文词典未覆盖、已丢弃的片段：{leftover}"
                                    f"（如影响结果，可开 --translate llm 用本地 qwen3:8b 整句翻译）")
                if not hit:
                    eq.notes.append("⚠ 中文查询里没有命中任何词典条目，翻译结果可能不可用")
            if out:
                eq.translated_from = src
                core = out
        elif eq.language in ("zh", "mixed"):
            eq.notes.append("⚠ 查询含中文但翻译已关闭；索引是英文模型 bge-base-en-v1.5，"
                            "直接检索中文会命中噪声")
        eq.core_text = core

        # 4) 识别医学实体
        eq.entities = self.extract_entities(eq.core_text)

        # 5) 同义词扩展
        eq.expansions = self.expand_synonyms(eq.entities)

        # 6) 生成不同的查询版本
        eq.vector_queries, eq.vector_query_weights = self._build_vector_queries(
            eq.core_text, eq.entities, eq.expansions)
        flat = [s for v in eq.expansions.values() for s in v]
        eq.vector_query_expanded = self.instr + (
            eq.core_text + (" (" + "; ".join(flat) + ")" if flat else ""))
        eq.keyword_groups, eq.keyword_query = self._build_keyword_query(
            eq.core_text, eq.entities, eq.expansions)

        if not eq.entities:
            eq.notes.append("未识别到医学实体：查询将按原文做稠密检索（不影响可用性，"
                            "只是没有同义词增益）")
        return eq


# 模块级便捷函数（任务书风格的调用方式）
_DEFAULT_PROCESSOR: Optional[MedicalQueryProcessor] = None


def process_query(query: str, **kw) -> EnhancedQuery:
    """处理医学查询，返回增强的查询信息（复用全局单例处理器）。"""
    global _DEFAULT_PROCESSOR
    if _DEFAULT_PROCESSOR is None:
        _DEFAULT_PROCESSOR = MedicalQueryProcessor(**kw)
    return _DEFAULT_PROCESSOR.process_query(query)


# ============================================================================
# 演示查询集：覆盖缩写/歧义/俗称/商品名/拼写/中文/时间过滤/章节过滤/无实体
# ============================================================================
DEMO_QUERIES = [
    "二甲双胍对心血管疾病有何影响？",
    "What is the effect of metformin on cardiovascular outcomes in T2DM?",
    "Does MI risk increase in patients with CKD?",
    "Is Glucophage safe for elderly patients with impaired renal function?",
    "heart attack prevention with aspirin, recent studies",
    "tumour haemorrhage after anti-VEGF therapy",
    "RCT evidence for pembrolizumab in NSCLC published since 2020",
    "EGFR mutation and TKI resistance in lung cancer, in the results section",
    "近五年 CRISPR 基因编辑在肿瘤治疗中的应用",
    "What did studies between 2015 and 2018 report about gut microbiota and obesity?",
    "How does CRISPR-Cas9 achieve targeted genome editing?",
    "hello world",
]


def dump_stats(out_path: str, mesh_dict_path: str = MESH_DICT,
               corpus_meta_path: str = CORPUS_META):
    """导出词典与语料统计（交付用）。"""
    st = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "static_dict": {
            "abbreviations": len(MEDICAL_ABBREVIATIONS),
            "ambiguous_abbreviations": sorted(
                k for k, v in MEDICAL_ABBREVIATIONS.items() if len(v) > 1),
            "lay_terms": len(MEDICAL_LAY_TERMS),
            "drug_brands": len(DRUG_BRAND_TO_GENERIC),
            "spelling_variants": len(SPELLING_VARIANTS),
            "cn_en_terms": len(CN_EN_MEDICAL),
            "total_entries": (len(MEDICAL_ABBREVIATIONS) + len(MEDICAL_LAY_TERMS)
                              + len(DRUG_BRAND_TO_GENERIC) + len(SPELLING_VARIANTS)
                              + len(CN_EN_MEDICAL)),
            "MEDICAL_SYNONYMS_merged": len(MEDICAL_SYNONYMS),
        },
        "entity_patterns": {
            "categories": list(MEDICAL_PATTERNS),
            "count": len(MEDICAL_PATTERNS),
        },
        "generic_noise_blocklist": len(_GENERIC_MEDICAL_NOISE),
    }
    if os.path.exists(mesh_dict_path):
        with open(mesh_dict_path, encoding="utf-8") as f:
            st["mesh_dict"] = json.load(f)["meta"]
    if os.path.exists(corpus_meta_path):
        with open(corpus_meta_path, encoding="utf-8") as f:
            cm = json.load(f)
        st["corpus_meta"] = {
            "rows": cm["meta"]["rows"],
            "pub_year": {k: cm["pub_year"][k] for k in ("min", "max", "p05")},
            "section_distinct_raw_values": cm["section"]["distinct_raw_values"],
            "section_canonical_counts": cm["section"]["canonical_counts"],
            "section_raw_variants_needed_for_99pct": {
                k: len(v) for k, v in cm["section"]["canonical_to_raw"].items()},
            "section_unmapped_pct": round(
                100 * cm["section"]["unmapped_total"]
                / (sum(cm["section"]["canonical_counts"].values())
                   + cm["section"]["unmapped_total"]), 1),
            "journal_distinct": cm["journal"]["distinct"],
        }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    print(json.dumps(st, ensure_ascii=False, indent=2))
    print(f"\n[统计] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=None, help="单条查询")
    ap.add_argument("--demo", action="store_true", help="跑内置演示查询集")
    ap.add_argument("--stats", default=None, metavar="OUT_JSON",
                    help="导出词典与语料统计到指定 JSON 后退出")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非可读文本")
    ap.add_argument("--translate", default="dict", choices=["off", "dict", "llm"])
    ap.add_argument("--instruction", default="official", choices=["official", "taskbook"])
    ap.add_argument("--no-mesh", action="store_true", help="关掉 MeSH 词典（仅用静态词典）")
    args = ap.parse_args()

    if args.stats:
        dump_stats(args.stats)
        return

    proc = MedicalQueryProcessor(
        use_mesh=not args.no_mesh,
        translate=args.translate,
        query_instruction=(QUERY_INSTRUCTION_OFFICIAL if args.instruction == "official"
                           else QUERY_INSTRUCTION_TASKBOOK),
        verbose=True,
    )

    qs = DEMO_QUERIES if args.demo else [args.query or DEMO_QUERIES[0]]
    for i, q in enumerate(qs, 1):
        eq = proc.process_query(q)
        if args.json:
            print(json.dumps(eq.to_dict(), ensure_ascii=False, indent=2))
        else:
            print("\n" + "=" * 78)
            print(f"【查询 {i}/{len(qs)}】")
            print("=" * 78)
            print(eq.pretty())


if __name__ == "__main__":
    main()
