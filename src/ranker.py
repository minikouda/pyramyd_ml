
def hybrid_score(results: list, priorities: dict) -> list:
    """
    Adjusts semantic search scores based on structured priorities.
    priorities: e.g. {"salary": 0.5, "rating": 0.3}
    """
    scored_results = []
    for r in results:
        # Start with semantic score (assumed normalized or cosine sim)
        final_score = r["score"]
        meta = r["meta"]
        
        # Boost by Rating
        if "rating" in priorities and meta.get("rating"):
            # Normalize rating 0-5 -> 0-1 approx
            norm_rating = meta["rating"] / 5.0
            final_score += priorities["rating"] * norm_rating
            
        # Boost by Salary
        if "salary" in priorities and meta.get("salary_median"):
            # Log scale salary to dampen massive outliers
            # Example assumption: 100k is baseline
            salary = meta["salary_median"]
            if salary > 0:
                import math
                norm_salary = math.log10(salary) / 6.0 # log10(100k)=5, log10(1m)=6
                final_score += priorities["salary"] * norm_salary

        r["hybrid_score"] = final_score
        scored_results.append(r)
        
    # Sort by new score
    return sorted(scored_results, key=lambda x: x["hybrid_score"], reverse=True)
