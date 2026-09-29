from pathlib import Path

from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_core.documents import Document

from ingest.load_pdf import DATA_DIR, load_sources

AUDIO_DIR = DATA_DIR / "raw" / "audio"


def transcribe(src):
    save_dir = AUDIO_DIR / src["id"]
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_file = save_dir / "transcript.txt"

    # 1. Use cached transcript if we already did this video
    if cache_file.exists():
        print(f"Using cached transcript for {src['id']}")
        return cache_file.read_text(encoding="utf-8")

    # 2. Blob loader: download from YouTube once, then reuse the local file
    if list(save_dir.glob("*.m4a")):
        blob_loader = FileSystemBlobLoader(str(save_dir), glob="*.m4a")
    else:
        print(f"Downloading audio for {src['id']}...")
        blob_loader = YoutubeAudioLoader([src["url"]], str(save_dir))

    # 3. Parser: local Whisper turns audio into text
    parser = FasterWhisperParser(device="cpu", model_size="base")
    loader = GenericLoader(blob_loader, parser)

    print(f"Transcribing {src['id']} (this can take a few minutes)...")
    parts = loader.load()
    text = " ".join(p.page_content for p in parts)

    cache_file.write_text(text, encoding="utf-8")
    return text


def load_videos():
    sources = load_sources()
    docs = []

    for src in sources.get("videos", []):
        try:
            text = transcribe(src)
        except Exception as e:
            print(f"FAILED: {src['url']} ({e})")
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source": src["url"],
                "topic": src["topic"],
                "agency": src["agency"],
                "source_type": "video",
                "source_url": src["url"],
                "last_checked": str(src["last_checked"]),
            },
        ))
        print(f"Loaded {src['id']} ({len(text)} chars)")

    return docs


if __name__ == "__main__":
    docs = load_videos()
    print(f"\nTotal videos: {len(docs)}")
    for doc in docs:
        print("\n" + "=" * 80)
        print(doc.metadata["source_url"])
        print("-" * 80)
        print(doc.page_content[:500])