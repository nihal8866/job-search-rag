"""Rebuild cleaned jobs, contextual chunks, embeddings, and a Chroma index."""

from pathlib import Path

from bs4 import BeautifulSoup
import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from job_rag.config import Settings


SOURCE_COLUMNS = [
    "ID", "Job Category", "Job Title", "Company Name", "Publication Date",
    "Job Location", "Job Level", "Tags", "Job Description",
]
METADATA_COLUMNS = [column for column in SOURCE_COLUMNS if column != "Job Description"]


def clean_description(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)


def prepare_jobs(dataset_path: Path) -> pd.DataFrame:
    """Audit required fields and preserve source values beside cleaned text."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    jobs = pd.read_excel(dataset_path)
    missing = set(SOURCE_COLUMNS) - set(jobs.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(sorted(missing))}")
    jobs = jobs[SOURCE_COLUMNS].copy()
    if jobs.empty:
        raise ValueError("The dataset contains no jobs.")
    required_text = [
        "ID", "Job Category", "Job Title", "Company Name", "Job Level", "Job Description"
    ]
    for column in required_text:
        valid = jobs[column].map(lambda value: isinstance(value, str) and bool(value.strip()))
        if not valid.all():
            raise ValueError(f"Missing or invalid text in {column}.")
    if not jobs["ID"].is_unique:
        raise ValueError("Job IDs must be unique.")
    dates = pd.to_datetime(jobs["Publication Date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("Publication dates contain missing or unparseable values.")

    jobs["Clean Description"] = jobs["Job Description"].apply(clean_description)
    if jobs["Clean Description"].str.strip().eq("").any():
        raise ValueError("HTML cleaning produced an empty description.")
    return jobs


def chunk_jobs(jobs: pd.DataFrame, tokenizer, settings: Settings, max_length: int) -> pd.DataFrame:
    """Use the notebook's token windows and contextual input format."""
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("Require 0 <= chunk overlap < chunk size.")
    step = settings.chunk_size - settings.chunk_overlap
    records = []
    for _, job in jobs.iterrows():
        text = job["Clean Description"]
        encoded = tokenizer(
            text, add_special_tokens=False, truncation=False,
            return_offsets_mapping=True, verbose=False,
        )
        offsets = encoded["offset_mapping"]
        if not offsets:
            raise ValueError(f"Job {job['ID']} has no description tokens.")
        header = (
            f"passage: Job title: {job['Job Title']}\n"
            f"Company: {job['Company Name']}\n"
            f"Category: {job['Job Category']}\n"
            f"Level: {job['Job Level']}\n\n"
        )
        for chunk_index, start in enumerate(range(0, len(offsets), step)):
            end = min(start + settings.chunk_size, len(offsets))
            chunk_text = text[offsets[start][0]:offsets[end - 1][1]]
            record = job[METADATA_COLUMNS].to_dict()
            record.update({
                "chunk_id": f"{job['ID']}_chunk_{chunk_index:03d}",
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "embedding_text": header + chunk_text,
                "content_token_count": len(tokenizer.encode(
                    chunk_text, add_special_tokens=False, truncation=False, verbose=False
                )),
                "metadata_token_count": len(tokenizer.encode(
                    header, truncation=False, verbose=False
                )),
                "input_token_count": len(tokenizer.encode(
                    header + chunk_text, truncation=False, verbose=False
                )),
            })
            if not chunk_text.strip():
                raise ValueError(f"Empty text in chunk {record['chunk_id']}.")
            if record["input_token_count"] > max_length:
                raise ValueError(f"Embedding input exceeds the model limit: {record['chunk_id']}")
            records.append(record)
            if end == len(offsets):
                break

    chunks = pd.DataFrame(records)
    if not chunks["chunk_id"].is_unique or set(chunks["ID"]) != set(jobs["ID"]):
        raise RuntimeError("Chunk identifiers or job coverage are inconsistent.")
    return chunks


def ingest_dataset(dataset_path: Path, settings: Settings) -> dict[str, object]:
    """Prepare and persist artifacts; reject stale IDs rather than mixing corpora."""
    if settings.embedding_batch_size < 1 or settings.ingestion_batch_size < 1:
        raise ValueError("Embedding and ingestion batch sizes must be positive.")
    jobs = prepare_jobs(dataset_path)
    print(f"Prepared {len(jobs)} jobs", flush=True)
    model = SentenceTransformer(settings.embedding_model, device=settings.device)
    chunks = chunk_jobs(jobs, model.tokenizer, settings, model.max_seq_length)
    print(f"Encoding {len(chunks)} chunks", flush=True)
    embeddings = model.encode(
        chunks["embedding_text"].tolist(),
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    expected_shape = (len(chunks), model.get_embedding_dimension())
    if embeddings.shape != expected_shape or not np.isfinite(embeddings).all():
        raise RuntimeError("Invalid embedding array shape or values.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Embeddings are not normalized.")

    settings.database_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.database_path))
    collection = client.get_or_create_collection(
        name=settings.collection_name, embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": settings.embedding_model},
    )
    if (collection.metadata or {}).get("embedding_model") != settings.embedding_model:
        raise ValueError("The existing collection uses a different embedding model.")
    if collection.configuration["hnsw"]["space"] != "cosine":
        raise ValueError("The existing collection must use cosine distance.")
    expected_ids = chunks["chunk_id"].tolist()
    stale_ids = set(collection.get(include=[])["ids"]) - set(expected_ids)
    if stale_ids:
        raise ValueError("The collection contains stale IDs. Use a fresh project root/database.")
    metadata = []
    for _, row in chunks.iterrows():
        item = {column: str(row[column]) for column in METADATA_COLUMNS if pd.notna(row[column])}
        item["chunk_index"] = int(row["chunk_index"])
        metadata.append(item)
    for start in range(0, len(chunks), settings.ingestion_batch_size):
        end = min(start + settings.ingestion_batch_size, len(chunks))
        collection.upsert(
            ids=expected_ids[start:end], embeddings=embeddings[start:end].tolist(),
            documents=chunks["chunk_text"].iloc[start:end].tolist(),
            metadatas=metadata[start:end],
        )
    if set(collection.get(include=[])["ids"]) != set(expected_ids):
        raise RuntimeError("Stored IDs do not match generated chunks.")

    output_dir = settings.chunks_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / "jobs_cleaned.jsonl"
    embeddings_path = output_dir / "job_embeddings.npy"
    jobs.to_json(cleaned_path, orient="records", lines=True, force_ascii=False)
    chunks.to_json(settings.chunks_path, orient="records", lines=True, force_ascii=False)
    np.save(embeddings_path, embeddings, allow_pickle=False)

    reloaded = pd.read_json(settings.chunks_path, lines=True, convert_dates=False)
    pd.testing.assert_frame_equal(chunks.isna(), reloaded.isna())
    pd.testing.assert_frame_equal(chunks.fillna(""), reloaded.fillna(""))
    np.testing.assert_array_equal(embeddings, np.load(embeddings_path, allow_pickle=False))
    return {
        "jobs": len(jobs), "chunks": len(chunks),
        "embedding_dimensions": int(embeddings.shape[1]),
        "maximum_input_tokens": int(chunks["input_token_count"].max()),
        "chunks_path": str(settings.chunks_path),
        "embeddings_path": str(embeddings_path),
        "database_path": str(settings.database_path),
    }
