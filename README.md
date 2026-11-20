# Effect of incorrect knowledge base values on retrieval-augmented radiology question answering

Yangtingting Wang, Saba Afza, Sven Nebelung, Andreas Schuppert, Daniel Truhn, and Soroosh Tayebi Arasteh

This repository contains the code of the study. A multi-step retrieval-augmented generation (RAG) pipeline writes an evidence report from a local copy of Radiopaedia articles for each of the 200 questions of RadSaFE-200. Seven open language models answer each question with its report or with evidence given directly. The reports and the answers are compared across the following conditions, named as in `config.yaml`:

- Numerical values changed in 10%, 25%, 50%, or 100% of the articles of the knowledge base (`num010` to `num100`)
- Removal of 10%, 25%, 50%, or 100% of the articles retrieved for the reports of the unchanged knowledge base (`cov010` to `cov100`), and removal of the article cited most often in each report (`top1_100`)
- Retrieval of 1, 3, 5, 10, or 20 articles per query, and of 5 articles with a share of 50%, 75%, 90%, or 95% irrelevant articles (`topk01_pad00` to `topk05_pad95`)
- Repeated runs on the unchanged knowledge base (`clean_r2`, `clean_r3`) and reports written without retrieved articles (`noretrieval`)
- Two further report models (`dsv4_*`, `q35_*`)
- Answers without evidence (`closedbook`) and with the evidence of the benchmark in its clean and conflict versions, with changed numbers, and rewritten in four tones (`ev_*`)
- Direct retrieval of 5 articles without a report model, from the unchanged knowledge base and from the knowledge base with all values changed (`rag_clean`, `rag_num100`)

## Installation and configuration

Clone the repository and install the packages that the modules import: LangChain (`langchain-core`, `langchain-openai`, `langchain-community`, `langchain-groq`, and `langchain-ollama`), LangGraph, LangSmith, Pydantic, python-dotenv, httpx, rank-bm25, NumPy, SciPy, pandas, and PyYAML.

Every path in `config.yaml` that starts with `/PATH` is a placeholder for a folder on your machine. The report models, the answering models, and the models that rewrite and check the evidence are served by an OpenAI-compatible server. Export its address and its key in the environment variables named in the `api` section of `config.yaml` (`CUSTOM_API_URL` and `CUSTOM_API_KEY`).

RadSaFE-200 is available on Hugging Face at https://huggingface.co/datasets/soroosharasteh/RadSaFE-200 (`benchmark.source_csv`). The pipeline also reads its questions as a JSON list (`benchmark.runtime_json`) with one object per question and the fields `question_id`, `q_number`, `dataset`, `summary` (the question text), `options` (the option texts under the letters A to E), `answer_index`, `question_type`, `subspecialty`, `source_url`, and `source_case_title`. The Radiopaedia articles are available on request from Radiopaedia at https://radiopaedia.org (`corpora.clean`, one text file per article). Neither source is redistributed in this repository. The module `src/result_tables.py` also reads the labels of the answer options by a second radiologist from the file in `result_tables.second_reader_csv`. These labels are based on the study of RadSaFE-200 (Wind et al., "Safety and accuracy diverge across deployment conditions in clinical large language models"). This file has the columns `dataset`, `q_number`, and `option_<k>_high_risk` for each option `k`.

## Code structure

Each entry function takes the path of `config.yaml` and is imported from the root of the repository, for example `from src.result_tables import main_result_tables`.

| Analysis | Implementation |
|---|---|
| Benchmark CSV converted to the runtime question JSON, with question and option validation | `src.result_tables.main_benchmark` |
| Numerical values changed in a share of the articles, with the log of every change | `scripts.numerical_perturbation.main_numerical_perturbation`, with the detection of numerical expressions in `scripts/numeric_patterns.py` |
| Retrieved articles removed | `src.rebuild_coverage_corpora.main_coverage_corpora` |
| Article cited most often in each report removed | `src.source_ablation_corpus.main_top1_corpus`, with the resolution of the cited sources in `src/cited_article_pool.py` |
| Clean evidence of the benchmark with changed numbers | `src.evidence_numeric.main_changed_evidence` |
| Rewrites of the clean evidence in four tones, with the number check and the content check of each attempt | `src.rewrite_benchmark_evidence.main_tone_rewrites` |
| Second content check of the final rewrites, which decides the accepted rewrites | `src.recheck_style_content_2.main_content_check` |
| Evidence reports of one condition | `src.report_generation.main_reports(config_path, condition)`, with the report pipeline in `rar_pipeline/` and the search without retrieved articles in `src/noretrieval.py` |
| Direct retrieval of 5 articles per question | `src.direct_rag.main_direct_retrieval` |
| Answers of one answering model in a list of conditions | `src.evaluate_answers.main_answers(config_path, model_name, conditions)`, with the parser of the option letter and the confidence in `src/parsing.py` |
| Changed and original values found in the reports | `src.analyze_numerical_matching.main_numerical_matching` |
| Table of all answers, outcomes of each answering model, and paired differences between conditions, with the labels of both radiologists | `src.result_tables.main_result_tables` |
| Replaced values in the reports, answers in the questions with a replaced value, means over the answering models, questions with changed evidence and with accepted rewrites, stated confidence, description of the reports and of the changed knowledge bases, changed values counted at their exact positions in the passages of direct retrieval, and a sensitivity analysis restricted to the article prefixes given to the report models | `src.propagation_tables.main_propagation_tables` |
| Counts of the answers, of the answers that changed with the changed evidence, of the logged changes that are a part of a larger number, and of the matches inside a ratio | `src.reported_counts.main_reported_counts` |

The functions that calculate bootstrap means and intervals, exact McNemar tests for paired binary outcomes, paired bootstrap tests for other outcomes, and correction for the false discovery rate are in `Inference/`.

## Citation

If you use this repository, please cite our paper:

```bibtex
@article{wang2026effect,
  title  = {Effect of incorrect knowledge base values on retrieval-augmented radiology question answering},
  author = {Wang, Yangtingting and Afza, Saba and Nebelung, Sven and Schuppert, Andreas and Truhn, Daniel and Tayebi Arasteh, Soroosh},
  year   = {2026}
}
```

## License

The code is released under the MIT license in `LICENSE`. The files in `rar_pipeline/` keep the MIT license of LangChain in `rar_pipeline/LICENSE.lanchain`.
