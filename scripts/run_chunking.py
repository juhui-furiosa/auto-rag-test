import argparse
import shutil
from pathlib import Path

from autorag.chunker import Chunker


def main():
	parser = argparse.ArgumentParser(description="Run AutoRAG chunker and create corpus.parquet.")
	parser.add_argument(
		"--parsed-data-path",
		type=Path,
		default=Path("raw.parquet"),
		help="Path to the raw.parquet generated from JSON data.",
	)
	parser.add_argument(
		"--chunk-config",
		type=Path,
		default=Path("configs/chunk_config.yaml"),
		help="Chunk YAML configuration file.",
	)
	parser.add_argument(
		"--work-dir",
		type=Path,
		default=Path("artifacts/chunk"),
		help="Directory to store intermediate chunk outputs.",
	)
	parser.add_argument(
		"--corpus-output",
		type=Path,
		default=Path("corpus.parquet"),
		help="Final corpus parquet output file.",
	)
	args = parser.parse_args()

	args.work_dir.mkdir(parents=True, exist_ok=True)

	chunker = Chunker.from_parquet(
		parsed_data_path=str(args.parsed_data_path),
		project_dir=str(args.work_dir),
	)
	chunker.start_chunking(str(args.chunk_config))

	first_chunk = args.work_dir / "0.parquet"
	if not first_chunk.exists():
		raise FileNotFoundError(
			f"{first_chunk} not found. Check chunk config for module definitions."
		)
	args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy(first_chunk, args.corpus_output)
	print(f"Copied {first_chunk} -> {args.corpus_output}")


if __name__ == "__main__":
	main()
