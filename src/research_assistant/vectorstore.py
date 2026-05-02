from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from research_assistant.config import CORPUS_DIR, Settings
from research_assistant.llm import build_embeddings


TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=18000,
    chunk_overlap=800,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def list_pdf_files(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    return sorted(corpus_dir.glob("*.pdf"))


def load_pdf_catalog(corpus_dir: Path = CORPUS_DIR) -> list[dict[str, str | int]]:
    catalog: list[dict[str, str | int]] = []
    for path in list_pdf_files(corpus_dir):
        reader = PdfReader(str(path))
        first_page_text = (reader.pages[0].extract_text() or "").strip() if reader.pages else ""
        title = ""
        metadata_title = getattr(reader.metadata, "title", None) if reader.metadata else None
        if metadata_title:
            title = str(metadata_title).strip()
        if not title:
            title = first_page_text.splitlines()[0].strip() if first_page_text else path.stem
        catalog.append(
            {
                "source": path.name,
                "title": title,
                "page_count": len(reader.pages),
                "preview": first_page_text[:600],
            }
        )
    return catalog


def _build_page_windows(
    pages: list[tuple[int, str]],
    *,
    max_chars: int = 50000,
    max_pages: int = 80,
) -> list[tuple[int, int, str]]:
    windows: list[tuple[int, int, str]] = []
    current_pages: list[int] = []
    current_parts: list[str] = []
    current_length = 0
    for page_number, text in pages:
        if current_pages and (
            current_length + len(text) > max_chars or len(current_pages) >= max_pages
        ):
            windows.append(
                (current_pages[0], current_pages[-1], "\n\n".join(current_parts).strip())
            )
            current_pages = []
            current_parts = []
            current_length = 0
        current_pages.append(page_number)
        current_parts.append(f"[Page {page_number}]\n{text}")
        current_length += len(text)
    if current_pages:
        windows.append(
            (current_pages[0], current_pages[-1], "\n\n".join(current_parts).strip())
        )
    return windows


def load_corpus_documents(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in list_pdf_files(corpus_dir):
        reader = PdfReader(str(path))
        title = path.stem
        if reader.metadata and getattr(reader.metadata, "title", None):
            title = str(reader.metadata.title).strip()
        extracted_pages: list[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = (page.extract_text() or "").strip()
            if not raw_text:
                continue
            extracted_pages.append((page_number, raw_text))

        for idx, (page_start, page_end, window_text) in enumerate(
            _build_page_windows(extracted_pages)
        ):
            page_title = window_text.splitlines()[0].strip() if window_text else title
            for sub_idx, chunk in enumerate(TEXT_SPLITTER.split_text(window_text)):
                if page_start == page_end:
                    page_reference = f"p. {page_start}"
                    chunk_id = f"{path.stem}-page-{page_start}-chunk-{sub_idx}"
                else:
                    page_reference = f"pp. {page_start}-{page_end}"
                    chunk_id = (
                        f"{path.stem}-pages-{page_start}-{page_end}-chunk-{sub_idx}"
                    )
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": path.name,
                            "title": title or page_title,
                            "page_number": page_start,
                            "page_end": page_end,
                            "page_reference": page_reference,
                            "chunk_id": chunk_id,
                        },
                    )
                )
    return documents


def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def recreate_vectorstore(settings: Settings) -> QdrantVectorStore:
    client = get_qdrant_client(settings)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    embedding = build_embeddings(settings)
    docs = load_corpus_documents()
    if not docs:
        raise ValueError(f"No PDF files found in {CORPUS_DIR}")
    vector_size = len(embedding.embed_query("dimension probe"))
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embedding,
    )
    store.add_documents(docs)
    return store


def ensure_vectorstore(settings: Settings) -> QdrantVectorStore:
    client = get_qdrant_client(settings)
    embedding = build_embeddings(settings)
    if not client.collection_exists(settings.qdrant_collection):
        docs = load_corpus_documents()
        if not docs:
            raise ValueError(f"No PDF files found in {CORPUS_DIR}")
        vector_size = len(embedding.embed_query("dimension probe"))
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=embedding,
        )
        store.add_documents(docs)
        return store

    if client.count(settings.qdrant_collection, exact=True).count == 0:
        raise ValueError(
            f"Qdrant collection '{settings.qdrant_collection}' exists but is empty. "
            "Populate it with scripts/build_index.py or point QDRANT_URL "
            "to a populated collection."
        )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embedding,
    )
