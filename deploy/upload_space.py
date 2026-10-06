"""Upload the web app and the vector index to the Hugging Face Space.

Run from the repo root, after `hf auth login` (with a Write token):

    python -m deploy.upload_space

Only the files the app needs are uploaded, in one commit. The Space then
rebuilds the Docker image by itself. The Gemini key is NOT uploaded: it is
set as the GOOGLE_API_KEY secret in the Space settings.
"""
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO_ID = "kZarT/sg-newcomer-guide"
ROOT = Path(__file__).resolve().parent.parent

# path in the Space -> local path
FILES = {
    "Dockerfile": "deploy/Dockerfile",
    "requirements.txt": "deploy/requirements-app.txt",
    "README.md": "deploy/space_README.md",
    "app.py": "app.py",
    "ingest/__init__.py": "ingest/__init__.py",
    "ingest/load_pdf.py": "ingest/load_pdf.py",
    "data/sources.yaml": "data/sources.yaml",
}
FILES |= {f"rag/{p.name}": f"rag/{p.name}" for p in (ROOT / "rag").glob("*.py")}
# The index built locally with `python -m rag.vectorstore`.
FILES |= {p.relative_to(ROOT).as_posix(): p.relative_to(ROOT).as_posix()
          for p in (ROOT / "data" / "chroma").rglob("*") if p.is_file()}

# Never upload these, even if a pattern above matches them by mistake.
FORBIDDEN = (".env",)


def main():
    for name in FILES:
        if name.endswith(FORBIDDEN):
            raise SystemExit(f"Refusing to upload {name}")
    if not any(name.startswith("data/chroma/") for name in FILES):
        raise SystemExit("No index in data/chroma - run `python -m rag.vectorstore` first")

    api = HfApi()
    existing = api.list_repo_files(REPO_ID, repo_type="space")
    # Remove files we don't upload any more (e.g. the template's src/).
    stale = [f for f in existing if f not in FILES and f != ".gitattributes"]

    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(ROOT / local))
                  for name, local in FILES.items()]
    operations += [CommitOperationDelete(path_in_repo=f) for f in stale]

    print(f"Uploading {len(FILES)} files, deleting {len(stale)}: {stale}")
    commit = api.create_commit(REPO_ID, repo_type="space", operations=operations,
                               commit_message="Deploy app and vector index")
    print(f"Done: {commit.commit_url}")
    print(f"App: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
