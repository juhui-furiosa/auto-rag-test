import argparse
import os
from pathlib import Path

import pandas as pd
from llama_index.llms.openai import OpenAI

from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (
	make_basic_gen_gt,
	make_concise_gen_gt,
)
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
from autorag.data.qa.schema import Corpus, Raw


def main():
	parser = argparse.ArgumentParser(description="Generate qa.parquet from corpus.parquet using AutoRAG utilities.")
	parser.add_argument(
		"--raw-path",
		type=Path,
		default=Path("raw.parquet"),
		help="Path to the raw.parquet file.",
	)
	parser.add_argument(
		"--corpus-path",
		type=Path,
		default=Path("corpus.parquet"),
		help="Path to the chunked corpus parquet file.",
	)
	parser.add_argument(
		"--qa-output",
		type=Path,
		default=Path("qa.parquet"),
		help="Destination parquet for QA pairs.",
	)
	parser.add_argument(
		"--corpus-output",
		type=Path,
		default=Path("corpus.parquet"),
		help="Where to save the (possibly updated) corpus parquet. Defaults to input path.",
	)
	parser.add_argument(
		"--samples",
		type=int,
		default=200,
		help="Number of passages to sample for QA generation.",
	)
	parser.add_argument(
		"--llm-model",
		type=str,
		default="gpt-4o-mini",
		help="OpenAI model used for query/answer generation.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Temperature passed to the OpenAI model.",
	)
	parser.add_argument(
		"--openai-api-key",
		type=str,
		default=None,
		help="Optional API key override. Uses OPENAI_API_KEY env if omitted.",
	)
	args = parser.parse_args()

	if args.openai_api_key is None:
		args.openai_api_key = os.getenv("OPENAI_API_KEY")
	if not args.openai_api_key:
		raise EnvironmentError("OPENAI_API_KEY is required to run QA generation.")

	raw_df = pd.read_parquet(args.raw_path)
	corpus_df = pd.read_parquet(args.corpus_path)
	if corpus_df.empty:
		raise ValueError("Corpus dataframe is empty. Run chunking first.")

	sample_size = min(args.samples, len(corpus_df))
	if sample_size == 0:
		raise ValueError("Corpus dataframe does not have enough rows for sampling.")

	llm = OpenAI(model=args.llm_model, temperature=args.temperature, api_key=args.openai_api_key)
	raw_instance = Raw(raw_df)
	corpus_instance = Corpus(corpus_df, raw_instance)

	qa = (
		corpus_instance.sample(random_single_hop, n=sample_size)
		.map(lambda df: df.reset_index(drop=True))
		.make_retrieval_gt_contents()
		.batch_apply(factoid_query_gen, llm=llm, lang="en")
		.batch_apply(make_basic_gen_gt, llm=llm)
		.batch_apply(make_concise_gen_gt, llm=llm)
		.filter(dontknow_filter_rule_based, lang="en")
	)

	args.qa_output.parent.mkdir(parents=True, exist_ok=True)
	args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
	qa.to_parquet(str(args.qa_output), str(args.corpus_output))
	print(f"Saved QA dataset to {args.qa_output} (samples: {len(qa.data)})")


if __name__ == "__main__":
	main()
