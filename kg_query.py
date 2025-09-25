import os
import re
from typing import List, Tuple, Dict, Any

from neo4j import GraphDatabase
from rapidfuzz import process, fuzz


def get_driver():
    uri = os.getenv("NEO4J_URI", "neo4j+s://be166b17.databases.neo4j.io")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "iyynrq8inhytJkyyyvfq6B6i8CwWraNjnvRXihPEUiE")
    return GraphDatabase.driver(uri, auth=(user, password))


def fetch_all_columns(session) -> List[str]:
    result = session.run("MATCH (c:Column) RETURN c.name AS name")
    return [r["name"] for r in result]


def fuzzy_find_columns(candidates: List[str], query_text: str, limit: int = 3, score_cutoff: int = 75) -> List[Tuple[str, int]]:
    # Try full query
    results = process.extract(query_text, candidates, scorer=fuzz.WRatio, limit=limit)
    strong = [(name, int(score)) for name, score, _ in results if score >= score_cutoff]
    if strong:
        return strong
    # Try noun-ish tokens (keep letters, digits, underscores and spaces)
    tokens = re.findall(r"[\w()%.\-]+", query_text)
    tokens = [t for t in tokens if len(t) > 2]
    scored: Dict[str, int] = {}
    for tok in tokens:
        for name, score, _ in process.extract(tok, candidates, scorer=fuzz.WRatio, limit=limit):
            scored[name] = max(scored.get(name, 0), int(score))
    return sorted([(n, s) for n, s in scored.items() if s >= score_cutoff], key=lambda x: x[1], reverse=True)[:limit]


def detect_intent(query_text: str) -> str:
    s = query_text.lower()
    # Order matters; more specific first
    if any(k in s for k in ["contributors", "parts of", "part of", "components of", "what makes up", "breakdown of"]):
        return "PARTS"
    if any(k in s for k in ["sum of", "total of", "aggregate of", "adds up to"]):
        return "SUM"
    if any(k in s for k in ["derived from", "how computed", "how calculated", "derivation of", "formula for"]):
        return "DERIVATION"
    if any(k in s for k in ["neighbors", "related to", "connected to", "links to", "show relations"]):
        return "NEIGHBORHOOD"
    if any(k in s for k in ["path between", "connection between", "shortest path", "link between"]):
        return "PATH"
    # comparison / decision / indirect evidence queries
    if any(k in s for k in ["compare", "better", "best", "which", "where", "suitable", "should i", "should we", "choose"]):
        return "EVIDENCE"
    # Default
    return "NEIGHBORHOOD"


def cypher_for_intent(intent: str, focus: str, second: str | None = None) -> Tuple[str, Dict[str, Any]]:
    if intent == "PARTS":
        # what contributes to target (incoming PART_OF or SUM_OF)
        query = (
            "MATCH (p:Column)-[r:PART_OF|SUM_OF]->(t:Column {name: $name}) "
            "RETURN p.name AS part, type(r) AS rel, t.name AS total"
        )
        return query, {"name": focus}
    if intent == "SUM":
        # totals that include this (outgoing PART_OF to some total; or this is a total with SUM_OF incoming)
        query = (
            "MATCH (x:Column {name: $name})-[r:PART_OF]->(t:Column) "
            "RETURN x.name AS component, type(r) AS rel, t.name AS total"
        )
        return query, {"name": focus}
    if intent == "DERIVATION":
        query = (
            "MATCH p=(s:Column)-[r:DERIVED_FROM*1..3]->(t:Column {name: $name}) "
            "RETURN p"
        )
        return query, {"name": focus}
    if intent == "PATH" and second:
        query = (
            "MATCH (a:Column {name: $a}), (b:Column {name: $b}), "
            "p = shortestPath((a)-[*..6]-(b)) RETURN p"
        )
        return query, {"a": focus, "b": second}
    if intent == "EVIDENCE":
        # Multi-hop neighborhood around focus (and optional second) capturing indirect relations
        if second:
            query = (
                "MATCH (a:Column {name: $a}), (b:Column {name: $b}) "
                "MATCH p1 = (a)-[:PART_OF|SUM_OF|DERIVED_FROM|BELONGS_TO|ATTRIBUTE_OF*1..3]-(x) "
                "MATCH p2 = (b)-[:PART_OF|SUM_OF|DERIVED_FROM|BELONGS_TO|ATTRIBUTE_OF*1..3]-(y) "
                "WITH collect(p1)+collect(p2) AS paths "
                "UNWIND paths AS p RETURN p"
            )
            return query, {"a": focus, "b": second}
        query = (
            "MATCH (a:Column {name: $a}) "
            "MATCH p = (a)-[:PART_OF|SUM_OF|DERIVED_FROM|BELONGS_TO|ATTRIBUTE_OF*1..3]-(x) "
            "RETURN p"
        )
        return query, {"a": focus}
    # Neighborhood default
    query = (
        "MATCH (n:Column {name: $name})-[r]-(m:Column) "
        "RETURN n.name AS src, type(r) AS rel, m.name AS dst"
    )
    return query, {"name": focus}


def answer_query(nl_query: str, max_rows: int = 200) -> Dict[str, Any]:
    driver = get_driver()
    try:
        with driver.session() as session:
            names = fetch_all_columns(session)
            # fuzzy find up to two targets
            matches = fuzzy_find_columns(names, nl_query, limit=5, score_cutoff=65)
            if not matches:
                return {"error": "No relevant columns found for the query.", "query": nl_query}

            intent = detect_intent(nl_query)

            focus = matches[0][0]
            second = matches[1][0] if len(matches) > 1 else None
            if intent == "PATH" and not second:
                # try extract a second best even with lower score
                lower = fuzzy_find_columns(names, nl_query, limit=2, score_cutoff=50)
                second = lower[1][0] if len(lower) > 1 else None

            cypher, params = cypher_for_intent(intent, focus, second)
            # For path-returning queries we can't tack on LIMIT inside string uniformly; add separately when tabular
            if " RETURN p" in cypher:
                result = session.run(cypher, params)
            else:
                result = session.run(cypher + " LIMIT $limit", {**params, "limit": max_rows})

            records = []
            for rec in result:
                # Handle path results or tabular results
                if "p" in rec.keys():
                    records.append({"path": rec["p"].__str__()})
                else:
                    records.append({k: rec[k] for k in rec.keys()})

            out = {
                "query": nl_query,
                "intent": intent,
                "focus": focus,
                "second": second,
                "matches": matches,
                "cypher": cypher,
                "params": params,
            }
            # Post-process to compute a simple relevance if we returned edges
            out["rows"] = records[:max_rows]
            return out
    finally:
        driver.close()


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "contributors to Ground Water Recharge (ham)_Total"
    out = answer_query(q)
    # Minimal pretty print
    print({k: v for k, v in out.items() if k not in {"rows"}})
    for row in out.get("rows", [])[:10]:
        print(row)


