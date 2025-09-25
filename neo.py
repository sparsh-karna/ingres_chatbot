from neo4j import GraphDatabase
import pandas as pd
import os

# Load CSV with relationships designed for Neo4j
df = pd.read_csv("columns_with_relationships.csv")

# Neo4j connection details (env first, fallback to existing values)
uri = "neo4j+s://be166b17.databases.neo4j.io"
user = "neo4j"
password = "iyynrq8inhytJkyyyvfq6B6i8CwWraNjnvRXihPEUiE"

ALLOWED_REL_TYPES = {"BELONGS_TO", "PART_OF", "SUM_OF", "DERIVED_FROM", "ATTRIBUTE_OF"}

driver = GraphDatabase.driver(uri, auth=(user, password))

def ensure_constraints(tx):
    tx.run("""
        CREATE CONSTRAINT IF NOT EXISTS FOR (c:Column) REQUIRE c.name IS UNIQUE
    """)

def upsert_column(tx, name, description):
    tx.run(
        """
        MERGE (c:Column {name: $name})
        SET c.description = $description
        """,
        name=name,
        description=description,
    )

def create_relationship(tx, src_name, rel_type, dst_name):
    # rel_type is validated before call; safe to inject as a label in Cypher
    query = f"""
        MATCH (src:Column {{name: $src_name}})
        MERGE (dst:Column {{name: $dst_name}})
        MERGE (src)-[:{rel_type}]->(dst)
    """
    tx.run(query, src_name=src_name, dst_name=dst_name)

with driver.session() as session:
    session.execute_write(ensure_constraints)

    # First create all Column nodes with descriptions
    for _, row in df.iterrows():
        name = str(row.get("column_name", "")).strip()
        description = str(row.get("description", "")).strip()
        if not name:
            continue
        session.execute_write(upsert_column, name, description)

    # Then create relationships
    for _, row in df.iterrows():
        name = str(row.get("column_name", "")).strip()
        relSpec = row.get("relationship")
        if not name or pd.isna(relSpec):
            continue
        relSpec = str(relSpec).strip()
        if not relSpec:
            continue

        # Parse "TYPE:Target|Target2"; split at first ':' only
        if ":" not in relSpec:
            continue
        rel_type, targets_spec = relSpec.split(":", 1)
        rel_type = rel_type.strip().upper()
        if rel_type not in ALLOWED_REL_TYPES:
            continue
        targets = [t.strip() for t in targets_spec.split("|") if t.strip()]
        for tgt in targets:
            session.execute_write(create_relationship, name, rel_type, tgt)

driver.close()