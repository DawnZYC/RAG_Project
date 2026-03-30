"""
评估脚本 — 测量检索质量和拒答准确率。

指标：
- Hit@k：top-k 结果中包含正确来源的比例
- MRR：正确来源排名的平均倒数
- 拒答率：系统主动拒绝回答的比例

使用方法：
    python -m evaluation.eval --qa-file evaluation/qa_set.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.vector_store import VectorStore, DEFAULT_TOP_K
from app.generation.generator import Generator
from app.models import RAGResponse


# ── 加载测试集 ─────────────────────────────────────────────────────────────────

def load_qa_set(path: Path) -> list[dict]:
    """
    测试集格式：
    [
      {
        "question": "文档里说了什么？",
        "expected_sources": ["report.pdf"],  # 正确答案应该来自哪个文件
        "should_refuse": false               # 是否应该触发拒答
      }
    ]
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 检索评估 ───────────────────────────────────────────────────────────────────

def evaluate_retrieval(
    vector_store: VectorStore,
    qa_set: list[dict],
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    计算 Hit@k 和 MRR。
    只评估 should_refuse=False 的问题（有答案的问题）。
    """
    answerable = [q for q in qa_set if not q.get("should_refuse", False)]

    hits = 0
    reciprocal_ranks = []
    top_scores = []

    for item in answerable:
        question         = item["question"]
        expected_sources = set(item.get("expected_sources", []))

        results = vector_store.search(question, top_k=top_k)

        if results:
            top_scores.append(results[0].score)

        # Hit@k：正确来源是否出现在 top-k 里
        retrieved_sources = {rc.chunk.source for rc in results}
        if expected_sources & retrieved_sources:
            hits += 1

        # MRR：正确来源第一次出现的排名倒数
        rr = 0.0
        for rc in results:
            if rc.chunk.source in expected_sources:
                rr = 1.0 / rc.rank
                break
        reciprocal_ranks.append(rr)

    n = len(answerable)
    return {
        "评估问题数":      n,
        "top_k":          top_k,
        f"Hit@{top_k}":   round(hits / n, 3) if n > 0 else 0,
        "MRR":            round(sum(reciprocal_ranks) / n, 3) if n > 0 else 0,
        "平均最高分数":    round(sum(top_scores) / len(top_scores), 3) if top_scores else 0,
    }


# ── 拒答评估 ───────────────────────────────────────────────────────────────────

def evaluate_refusal(
    vector_store: VectorStore,
    generator: Generator,
    qa_set: list[dict],
) -> dict:
    """
    检查拒答逻辑是否正确：
    - should_refuse=True 的问题，系统应该拒答
    - should_refuse=False 的问题，系统应该给出回答
    """
    correct = 0
    total   = len(qa_set)

    for item in qa_set:
        question      = item["question"]
        should_refuse = item.get("should_refuse", False)

        retrieved = vector_store.search(question)
        response: RAGResponse = generator.generate(question, retrieved)

        if response.refused == should_refuse:
            correct += 1

    return {
        "评估问题数":  total,
        "拒答准确率":  round(correct / total, 3) if total > 0 else 0,
    }


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="评估 RAG pipeline")
    parser.add_argument("--qa-file",    type=Path, required=True, help="测试集 JSON 文件路径")
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/chroma"))
    parser.add_argument("--top-k",      type=int,  default=DEFAULT_TOP_K)
    parser.add_argument("--skip-generation", action="store_true", help="只评估检索，跳过 LLM 调用")
    args = parser.parse_args()

    import os
    from dotenv import load_dotenv
    load_dotenv()

    print("加载向量库...")
    vs = VectorStore(persist_dir=args.chroma_dir)
    print(f"  当前 chunk 数量: {vs.count()}\n")

    qa_set = load_qa_set(args.qa_file)
    print(f"加载测试集，共 {len(qa_set)} 条问题\n")

    # 检索评估
    print("── 检索评估 ──────────────────────")
    retrieval_metrics = evaluate_retrieval(vs, qa_set, top_k=args.top_k)
    for k, v in retrieval_metrics.items():
        print(f"  {k}: {v}")

    # 生成 + 拒答评估
    if not args.skip_generation:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("\n跳过生成评估：OPENAI_API_KEY 未设置")
            return

        print("\n── 拒答评估 ──────────────────────")
        gen = Generator(api_key=api_key)
        refusal_metrics = evaluate_refusal(vs, gen, qa_set)
        for k, v in refusal_metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
