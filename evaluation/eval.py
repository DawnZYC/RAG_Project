"""
评估脚本 — 测量检索质量和拒答准确率。

支持通过 --backend 参数切换评估的检索后端（native / llamaindex），
或通过 --compare 同时跑两个后端并对比结果。

指标：
- Hit@k：top-k 结果中包含正确来源文件的比例
- MRR：正确来源第一次出现的排名倒数均值（越高说明正确结果排得越靠前）
- 拒答准确率：should_refuse 标注与系统实际拒答行为的吻合比例

使用方法：
    # 评估单个后端
    python -m evaluation.eval --qa-file evaluation/qa_set.json

    # 指定后端
    python -m evaluation.eval --qa-file evaluation/qa_set.json --backend llamaindex

    # 对比两个后端
    python -m evaluation.eval --qa-file evaluation/qa_set.json --compare

    # 只跑检索评估，跳过 LLM 调用（省 Token）
    python -m evaluation.eval --qa-file evaluation/qa_set.json --skip-generation
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 让脚本能直接作为模块运行，找到 app 包
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from app.retrieval.factory import create_vector_store
from app.retrieval.vector_store import DEFAULT_TOP_K
from app.generation.generator import Generator
from app.models import RAGResponse


# ── 测试集格式说明 ──────────────────────────────────────────────────────────────
#
# evaluation/qa_set.json 格式：
# [
#   {
#     "question": "文档里说了什么？",
#     "expected_sources": ["report.pdf"],   # 正确答案应该来自哪些文件
#     "should_refuse": false                 # true = 这道题应该触发拒答
#   },
#   {
#     "question": "火星上有多少人口？",
#     "expected_sources": [],
#     "should_refuse": true
#   }
# ]


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_qa_set(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"加载测试集：{path}，共 {len(data)} 条\n")
    return data


def build_vector_store(backend: str, chroma_dir: Path, embedding_model: str):
    """
    通过工厂函数创建检索后端。
    eval.py 与 main.py 用同一个 factory，保证行为一致。
    """
    print(f"  初始化 [{backend}] 后端...")
    vs = create_vector_store(
        backend=backend,
        persist_dir=chroma_dir,
        model_name=embedding_model,
    )
    print(f"  ChromaDB chunk 数量: {vs.count()}\n")
    return vs


# ── 检索评估 ───────────────────────────────────────────────────────────────────

def evaluate_retrieval(vector_store, qa_set: list[dict], top_k: int) -> dict:
    """
    计算 Hit@k 和 MRR。
    只统计 should_refuse=False 的问题（理论上有答案的问题）。

    Hit@k：top-k 结果中至少有一个 chunk 来自正确文件 → 计 1 分
    MRR  ：正确来源第一次出现的排名倒数（rank=1 → 1.0，rank=2 → 0.5，没出现 → 0）
    """
    answerable = [q for q in qa_set if not q.get("should_refuse", False)]
    if not answerable:
        return {"警告": "测试集中没有 should_refuse=False 的问题"}

    hits = 0
    reciprocal_ranks = []
    top_scores = []

    for item in answerable:
        question         = item["question"]
        expected_sources = set(item.get("expected_sources", []))

        results = vector_store.search(question, top_k=top_k)

        # 记录最高分，用于观察分数分布
        if results:
            top_scores.append(results[0].score)

        # Hit@k
        retrieved_sources = {rc.chunk.source for rc in results}
        if expected_sources & retrieved_sources:
            hits += 1

        # MRR：找正确来源第一次出现的位置
        rr = 0.0
        for rc in results:
            if rc.chunk.source in expected_sources:
                rr = 1.0 / rc.rank
                break
        reciprocal_ranks.append(rr)

    n = len(answerable)
    return {
        "有答案问题数":   n,
        "top_k":         top_k,
        f"Hit@{top_k}":  round(hits / n, 3),
        "MRR":           round(sum(reciprocal_ranks) / n, 3),
        "平均最高分数":   round(sum(top_scores) / len(top_scores), 3) if top_scores else 0,
    }


# ── 拒答评估 ───────────────────────────────────────────────────────────────────

def evaluate_refusal(vector_store, generator: Generator, qa_set: list[dict]) -> dict:
    """
    测试拒答逻辑是否正确，同时统计漏答和误答。

    正确拒答（True Positive） ：should_refuse=True  且系统拒答
    错误拒答（False Positive）：should_refuse=False 且系统拒答（本有答案但系统拒了）
    漏拒（False Negative）    ：should_refuse=True  且系统给了回答（本该拒但没拒）
    """
    tp = fp = fn = correct = 0
    total = len(qa_set)

    for item in qa_set:
        question      = item["question"]
        should_refuse = item.get("should_refuse", False)

        retrieved        = vector_store.search(question)
        response: RAGResponse = generator.generate(question, retrieved)

        if response.refused == should_refuse:
            correct += 1

        if should_refuse and response.refused:
            tp += 1
        elif not should_refuse and response.refused:
            fp += 1
        elif should_refuse and not response.refused:
            fn += 1

    return {
        "评估问题数":         total,
        "拒答准确率":         round(correct / total, 3),
        "正确拒答(TP)":      tp,
        "误拒答(FP，本有答案)": fp,
        "漏拒答(FN，应拒未拒)": fn,
    }


# ── 打印结果 ───────────────────────────────────────────────────────────────────

def print_metrics(title: str, metrics: dict):
    print(f"── {title} {'─' * max(0, 30 - len(title))}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()


def compare_backends(
    backends: list[str],
    chroma_dir: Path,
    embedding_model: str,
    qa_set: list[dict],
    top_k: int,
):
    """
    对比多个后端的检索质量，输出并排对比表。
    注意：两个后端必须用相同文档、相同 embedding 模型才有可比性。
    如果 embedding 模型不同，对比结果没有意义。
    """
    results = {}
    for backend in backends:
        print(f"正在评估 [{backend}] 后端...")
        vs = build_vector_store(backend, chroma_dir, embedding_model)
        metrics = evaluate_retrieval(vs, qa_set, top_k)
        results[backend] = metrics

    # 并排打印
    print(f"\n{'对比项':<20}", end="")
    for b in backends:
        print(f"[{b}]".ljust(18), end="")
    print()
    print("─" * (20 + 18 * len(backends)))

    keys = list(next(iter(results.values())).keys())
    for key in keys:
        print(f"{key:<20}", end="")
        for b in backends:
            val = str(results[b].get(key, "-"))
            print(val.ljust(18), end="")
        print()
    print()


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="评估 RAG pipeline 检索质量和拒答准确率")
    parser.add_argument(
        "--qa-file", type=Path, required=True,
        help="测试集 JSON 文件路径，格式见脚本顶部注释"
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["native", "llamaindex"],
        help="指定评估哪个后端（默认读取 .env 的 RAG_BACKEND）"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="同时评估 native 和 llamaindex 并对比（两个后端需用相同 embedding 模型）"
    )
    parser.add_argument(
        "--chroma-dir", type=Path, default=None,
        help="ChromaDB 路径（默认读取 .env 的 CHROMA_DIR）"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"检索返回数量（默认 {DEFAULT_TOP_K}）"
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="只跑检索评估，跳过 LLM 调用（节省 Token）"
    )
    args = parser.parse_args()

    load_dotenv()

    # 从环境变量读取配置，命令行参数优先
    chroma_dir      = args.chroma_dir or Path(os.getenv("CHROMA_DIR", "data/chroma"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    backend         = args.backend or os.getenv("RAG_BACKEND", "native").lower()

    qa_set = load_qa_set(args.qa_file)

    # ── 对比模式 ───────────────────────────────────────────────────────────────
    if args.compare:
        compare_backends(
            backends=["native", "llamaindex"],
            chroma_dir=chroma_dir,
            embedding_model=embedding_model,
            qa_set=qa_set,
            top_k=args.top_k,
        )
        return

    # ── 单后端模式 ─────────────────────────────────────────────────────────────
    print(f"评估后端：[{backend}]")
    vs = build_vector_store(backend, chroma_dir, embedding_model)

    # 检索评估
    retrieval_metrics = evaluate_retrieval(vs, qa_set, top_k=args.top_k)
    print_metrics("检索评估", retrieval_metrics)

    # 拒答评估（调用 OpenAI，可跳过）
    if not args.skip_generation:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("跳过拒答评估：OPENAI_API_KEY 未设置")
            return

        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        gen = Generator(api_key=api_key, model=llm_model)

        refusal_metrics = evaluate_refusal(vs, gen, qa_set)
        print_metrics("拒答评估", refusal_metrics)


if __name__ == "__main__":
    main()
