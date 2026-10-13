"""
Unmayaverse: Multimodal Retrieval Benchmark Evaluation
Evaluates search engine performance across MRR, Recall@K, and Negative Rejection
against the golden benchmark dataset.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

from src.config import (
    MEDIA_FOLDER_PATH,
    GOLDEN_DATASET_FILE,
    SEARCH_MODES,
    SEARCH_MODE_VISUAL,
    SEARCH_MODE_VISUAL_TEXT,
    SEARCH_MODE_VISUAL_TEXT_AUDIO,
)
from src.search_engine import MultimodalSearchEngine


def evaluate_retrieval(
    search_engine: MultimodalSearchEngine,
    dataset: List[Dict[str, Any]],
    mode: str = SEARCH_MODE_VISUAL_TEXT_AUDIO,
    k_values: List[int] = [1, 3, 5, 10],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluates search retrieval against ground truth golden dataset.
    
    Args:
        search_engine: Initialized MultimodalSearchEngine.
        dataset: List of query dictionaries with ground truth intervals.
        mode: Retrieval mode to evaluate ('visual', 'visual+text', 'visual+text+audio').
        k_values: Ranks to evaluate Recall@K at.
        verbose: If True, prints per-query rank logs.
        
    Returns:
        Dictionary containing MRR, Recall@K, Rejection Accuracy, and modality breakdowns.
    """
    max_k = max(k_values)
    mrr_sum = 0.0
    recall_counts = {k: 0 for k in k_values}
    
    pos_queries = 0
    neg_queries = 0
    neg_correct_rejections = 0
    
    # Modality breakdown tracking
    modality_stats: Dict[str, Dict[str, Any]] = {}
    detailed_records: List[Dict[str, Any]] = []

    for item in dataset:
        query_id = item.get("query_id", "unknown")
        query_text = item["query"]
        modality = item.get("modality_target", "Unknown")
        target_start = item.get("ground_truth_start")
        target_end = item.get("ground_truth_end")
        
        is_negative = (modality == "Negative") or (target_start is None) or (target_end is None)
        
        # Execute retrieval query
        results = search_engine.search(
            query_text=query_text,
            mode=mode,
            top_k=max_k,
        )
        
        # Negative / Out-of-Domain evaluation
        if is_negative:
            neg_queries += 1
            is_rejected = (len(results) == 0)
            if is_rejected:
                neg_correct_rejections += 1
            detailed_records.append({
                "query_id": query_id,
                "modality": modality,
                "query": query_text,
                "is_negative": True,
                "rejected": is_rejected,
                "retrieved_count": len(results)
            })
            if verbose:
                status = "PASS (Rejected)" if is_rejected else f"FAIL ({len(results)} false positives)"
                print(f"[{query_id}] [NEG] {query_text[:50]}... -> {status}")
            continue

        # Positive query evaluation
        pos_queries += 1
        retrieved_timestamps = [r["timestamp"] for r in results]
        
        # Find first rank satisfying ground truth time window
        first_hit_rank: Optional[int] = None
        for rank, ts in enumerate(retrieved_timestamps, 1):
            if target_start <= ts <= target_end:
                first_hit_rank = rank
                break
        
        # Reciprocal Rank calculation
        reciprocal_rank = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
        mrr_sum += reciprocal_rank
        
        # Recall@K calculation
        query_hits_at_k: Dict[int, bool] = {}
        for k in k_values:
            hit = any(target_start <= ts <= target_end for ts in retrieved_timestamps[:k])
            if hit:
                recall_counts[k] += 1
            query_hits_at_k[k] = hit
            
        # Accumulate stats by modality
        if modality not in modality_stats:
            modality_stats[modality] = {"count": 0, "mrr_sum": 0.0, "recall_5_count": 0}
        modality_stats[modality]["count"] += 1
        modality_stats[modality]["mrr_sum"] += reciprocal_rank
        if query_hits_at_k.get(5, False):
            modality_stats[modality]["recall_5_count"] += 1

        detailed_records.append({
            "query_id": query_id,
            "modality": modality,
            "query": query_text,
            "is_negative": False,
            "first_hit_rank": first_hit_rank,
            "reciprocal_rank": reciprocal_rank,
            "hits_at_k": query_hits_at_k
        })
        
        if verbose:
            hit_str = f"Hit @ Rank {first_hit_rank}" if first_hit_rank else "Missed"
            print(f"[{query_id}] [{modality:<10}] {query_text[:45]}... -> {hit_str}")

    # Compute overall aggregates
    mrr = (mrr_sum / pos_queries) if pos_queries > 0 else 0.0
    recall_at_k = {
        k: (recall_counts[k] / pos_queries if pos_queries > 0 else 0.0)
        for k in k_values
    }
    rejection_acc = (neg_correct_rejections / neg_queries * 100.0) if neg_queries > 0 else 0.0

    # Compute breakdown aggregates
    breakdown_summary = {}
    for mod, s in modality_stats.items():
        cnt = s["count"]
        breakdown_summary[mod] = {
            "count": cnt,
            "mrr": (s["mrr_sum"] / cnt) if cnt > 0 else 0.0,
            "recall@5": (s["recall_5_count"] / cnt) if cnt > 0 else 0.0
        }

    return {
        "mode": mode,
        "total_queries": len(dataset),
        "positive_queries": pos_queries,
        "negative_queries": neg_queries,
        "mrr": mrr,
        "recall_at_k": recall_at_k,
        "rejection_accuracy_pct": rejection_acc,
        "modality_breakdown": breakdown_summary,
        "detailed_queries": detailed_records
    }


def run_benchmark_comparison(
    search_engine: MultimodalSearchEngine,
    dataset: List[Dict[str, Any]],
    modes: List[str] = SEARCH_MODES,
    print_more_info: bool = False,
) -> List[Dict[str, Any]]:
    """Runs and tabulates comparative evaluation across multiple search modes."""
    results = []
    
    print("=" * 90)
    print(" UNMAYAVERSE MULTIMODAL RETRIEVAL BENCHMARK EVALUATION")
    print(f" Dataset Size: {len(dataset)} queries ")
    print("=" * 90)

    for mode in modes:
        print(f"\n[Evaluating Mode: {mode}]...")
        res = evaluate_retrieval(
            search_engine=search_engine,
            dataset=dataset,
            mode=mode,
            verbose=False
        )
        results.append(res)

        print("\n" + "=" * 90)
    print(f"{'Mode':<26} | {'MRR':<7} | {'R@1':<7} | {'R@3':<7} | {'R@5':<7} | {'R@10':<7} | {'Rejection':<10}")
    print("-" * 90)
    for r in results:
        r_k = r["recall_at_k"]
        print(
            f"{r['mode']:<26} | {r['mrr']:<7.4f} | {r_k[1]:<7.4f} | {r_k[3]:<7.4f} | "
            f"{r_k[5]:<7.4f} | {r_k[10]:<7.4f} | {r['rejection_accuracy_pct']:>6.1f}%"
        )
    print("=" * 90)

    if print_more_info:
        # 1. Print modality target breakdown for each evaluated mode
        for r in results:
            print(f"\nModality Target Breakdown for '{r['mode']}':")
            print("-" * 60)
            print(f"{'Target Modality':<20} | {'Queries':<8} | {'MRR':<8} | {'Recall@5':<8}")
            print("-" * 60)
            for mod, stat in r["modality_breakdown"].items():
                print(f"{mod:<20} | {stat['count']:<8} | {stat['mrr']:<8.4f} | {stat['recall@5']:<8.4f}")
            print("-" * 60)

        # 2. Print query text and first-hit rank for each mode
        for r in results:
            print("\n" + "=" * 90)
            print(f" DETAILED QUERY RANKINGS FOR MODE: '{r['mode']}'")
            print("=" * 90)
            print(f"{'ID':<10} | {'Target':<12} | {'Rank':<10} | Query Text")
            print("-" * 90)
            for q in r["detailed_queries"]:
                if q["is_negative"]:
                    rank_str = "REJECTED" if q["rejected"] else "FALSE POS"
                else:
                    rank_str = f"#{q['first_hit_rank']}" if q["first_hit_rank"] else "MISSED"
                print(f"{q['query_id']:<10} | {q['modality']:<12} | {rank_str:<10} | {q['query']}")
            print("-" * 90)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Unmayaverse Retrieval Pipeline")
    parser.add_argument("--media-folder", type=str, default=MEDIA_FOLDER_PATH, help="Path to media directory")
    parser.add_argument("--mode", type=str, default="all", choices=["all"] + SEARCH_MODES, help="Search mode")
    parser.add_argument("--verbose", action="store_true", help="Print per-query evaluation details")
    args = parser.parse_args()

    dataset_path = os.path.join(args.media_folder, GOLDEN_DATASET_FILE)
    if not os.path.exists(dataset_path):
        print(f"[Error] Golden dataset not found at: {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    engine = MultimodalSearchEngine(media_folder_path=args.media_folder)

    if args.mode == "all":
        run_benchmark_comparison(
            search_engine=engine,
            dataset=dataset,
        )
    else:
        res = evaluate_retrieval(
            search_engine=engine,
            dataset=dataset,
            mode=args.mode,
            verbose=args.verbose
        )
        print("\nEvaluation Results:")
        print(json.dumps({k: v for k, v in res.items() if k != "detailed_queries"}, indent=2))


if __name__ == "__main__":
    main()
