"""A vector store backed by Postgres with the pgvector extension."""

import math

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

from rag_research_agent.models import Chunk, SearchResult
from rag_research_agent.vectorstores.base import VectorStore


class PgVectorStore(VectorStore):
    """Stores chunks and vectors in a Postgres table, searched with pgvector.

    Unlike `InMemoryVectorStore`, the data survives restarts, can be shared by
    several processes, and an HNSW index keeps searches fast on many chunks.

    All vectors in one table must come from the same embedding model with the
    same dimensions: vectors from different models can't be compared. If the
    table already holds vectors of another size or model, creating the store
    raises ValueError instead of silently mixing them.
    """

    def __init__(
        self,
        database_url: str,
        dimensions: int,
        embedding_model: str,
        table: str = "chunks",
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._database_url = database_url
        self._dimensions = dimensions
        self._embedding_model = embedding_model
        self._table = sql.Identifier(table)
        self._create_schema(table)

    def _connect(self) -> psycopg.Connection:
        # A short-lived connection per call keeps things simple; pooling can come later
        conn = psycopg.connect(self._database_url, autocommit=True)
        register_vector(conn)
        return conn

    def _create_schema(self, table: str) -> None:
        with psycopg.connect(self._database_url, autocommit=True) as conn:
            # The extension must exist before register_vector can find the type
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id text PRIMARY KEY,
                        document_id text NOT NULL,
                        source text NOT NULL,
                        title text NOT NULL,
                        uri text NOT NULL,
                        "index" integer NOT NULL,
                        text text NOT NULL,
                        metadata jsonb NOT NULL DEFAULT '{{}}',
                        embedding vector({dimensions}) NOT NULL,
                        embedding_model text NOT NULL
                    )
                    """
                ).format(table=self._table, dimensions=sql.Literal(self._dimensions))
            )
            self._check_compatible(conn)
            conn.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {name} ON {table} (document_id)").format(
                    name=sql.Identifier(f"{table}_document_id_idx"), table=self._table
                )
            )
            # HNSW is an approximate nearest-neighbour index; vector_cosine_ops
            # makes it match the <=> (cosine distance) operator used in search()
            conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {name} ON {table} "
                    "USING hnsw (embedding vector_cosine_ops)"
                ).format(name=sql.Identifier(f"{table}_embedding_idx"), table=self._table)
            )

    def _check_compatible(self, conn: psycopg.Connection) -> None:
        # For a vector column, atttypmod holds its declared number of dimensions
        row = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = %s::regclass AND attname = 'embedding'",
            (self._table.as_string(conn),),
        ).fetchone()
        if row is None or row[0] != self._dimensions:
            found = "no embedding column" if row is None else f"{row[0]} dimensions"
            raise ValueError(
                f"Table {self._table.as_string(conn)} has {found}, but the embedder "
                f"produces {self._dimensions}. Use a different table or re-index "
                "everything with one embedding model."
            )
        row = conn.execute(
            sql.SQL(
                "SELECT embedding_model FROM {table} WHERE embedding_model <> %s LIMIT 1"
            ).format(table=self._table),
            (self._embedding_model,),
        ).fetchone()
        if row is not None:
            raise ValueError(
                f"Table {self._table.as_string(conn)} holds vectors from embedding model "
                f"{row[0]!r}, not {self._embedding_model!r}. Vectors from different models "
                "can't be compared; use a different table or re-index everything."
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors; they must match"
            )
        if not chunks:
            return
        query = sql.SQL(
            """
            INSERT INTO {table}
                (id, document_id, source, title, uri, "index", text, metadata,
                 embedding, embedding_model)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                source = EXCLUDED.source,
                title = EXCLUDED.title,
                uri = EXCLUDED.uri,
                "index" = EXCLUDED."index",
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model
            """
        ).format(table=self._table)
        rows = [
            (
                chunk.id,
                chunk.document_id,
                chunk.source,
                chunk.title,
                chunk.uri,
                chunk.index,
                chunk.text,
                Jsonb(chunk.metadata),
                Vector(vector),
                self._embedding_model,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        # One transaction, so a failing row leaves none of the batch half-written
        with self._connect() as conn, conn.transaction(), conn.cursor() as cur:
            cur.executemany(query, rows)

    def delete_document(self, document_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                sql.SQL("DELETE FROM {table} WHERE document_id = %s").format(table=self._table),
                (document_id,),
            )
            return cur.rowcount

    def search(
        self, query_vector: list[float], top_k: int = 5, source: str | None = None
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        where = sql.SQL("WHERE source = %(source)s") if source is not None else sql.SQL("")
        query = sql.SQL(
            """
            SELECT id, document_id, source, title, uri, "index", text, metadata,
                   embedding <=> %(query)s AS distance
            FROM {table}
            {where}
            ORDER BY distance
            LIMIT %(top_k)s
            """
        ).format(table=self._table, where=where)
        params = {"query": Vector(query_vector), "source": source, "top_k": top_k}
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            SearchResult(
                chunk=Chunk(
                    id=id,
                    document_id=document_id,
                    source=source,
                    title=title,
                    uri=uri,
                    index=index,
                    text=text,
                    metadata=metadata,
                ),
                # <=> is cosine distance (0 = same direction), so 1 - distance is
                # cosine similarity. An all-zero vector gives NaN; like
                # cosine_similarity, treat it as 0.0
                score=0.0 if math.isnan(distance) else 1.0 - distance,
            )
            for id, document_id, source, title, uri, index, text, metadata, distance in rows
        ]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                sql.SQL("SELECT count(*) FROM {table}").format(table=self._table)
            ).fetchone()
        assert row is not None
        return row[0]
