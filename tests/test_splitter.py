from rag_assistant.loaders import Document
from rag_assistant.splitter import chunk_documents, split_text


def test_short_text_returns_single_chunk():
    assert split_text("Короткий текст.", chunk_size=100, overlap=0) == ["Короткий текст."]


def test_empty_text_returns_empty_list():
    assert split_text("", chunk_size=100, overlap=10) == []
    assert split_text("   \n  ", chunk_size=100, overlap=10) == []


def test_long_text_is_split_by_paragraphs():
    text = "Абзац один.\n\nАбзац два.\n\nАбзац три.\n\nАбзац четыре."
    chunks = split_text(text, chunk_size=20, overlap=0)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.strip()


def test_overlap_creates_shared_tail():
    text = "AAAAAAAAAA\n\nBBBBBBBBBB\n\nCCCCCCCCCC"
    chunks = split_text(text, chunk_size=10, overlap=3)
    assert len(chunks) >= 2
    assert chunks[1].startswith(chunks[0][-3:])


def test_chunk_documents_preserves_metadata():
    docs = [Document(text="Привет, мир.", source="a.md", metadata={"page": 1})]
    chunks = chunk_documents(docs, chunk_size=50, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].source == "a.md"
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["chunk_index"] == 0
