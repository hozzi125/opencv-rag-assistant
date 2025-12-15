import argparse
from build_docs import build_corpus, load_sources
from index_chroma import build_chroma_index
from rag_pipeline import run_demo

def main():
  parser = argparse.ArgumentParser(
    description="RAG OpenCV Assistant: corpus builder, indexer and demo runner")

  parser.add_argument(
    "command",
    choices=["build-docs","build-index","demo"],
    help=(
      "build-docs -> завантажити документацію OpenCV та побудувати chunks.jsonl\n" 
      "build-index -> згенерувати embeddings та побудувати Chroma-індекс\n"
      "demo -> запустити RAG vs no-RAG демонстрацію"
  ),
  )

  args = parser.parse_args()

  if args.command == "build-docs":
    #корпус документів OpenCV
    sources = load_sources("sources.yaml")
    build_corpus(sources)

  elif args.command == "build-index":
    #embeddings + chroma + top-k retrieval
    build_chroma_index()
  
  elif args.command == "demo":
    #RAG пайплайн + порівняння з/без RAG
    run_demo()

if __name__ == "__main__":
  main()