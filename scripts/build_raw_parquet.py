import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import pandas as pd


def _iter_payloads(obj: Any) -> Iterator[Dict[str, Any]]:
	if isinstance(obj, dict):
		yield obj
	elif isinstance(obj, list):
		for item in obj:
			if isinstance(item, dict):
				yield item
			else:
				yield {"content": str(item)}
	else:
		yield {"content": str(obj)}


def _extract_text(payload: Dict[str, Any]) -> str:
	text = (
		payload.get("content")
		or payload.get("text")
		or payload.get("body")
		or payload.get("readme")
	)
	if text is None:
		text = "\n\n".join(_flatten_text(payload))
	elif not isinstance(text, str):
		text = "\n\n".join(_flatten_text(text))
	if not text.strip():
		raise ValueError("JSON payload does not contain textual fields.")
	return text


def _flatten_text(value: Any) -> Iterable[str]:
	if value is None:
		return []
	if isinstance(value, str):
		return [value]
	if isinstance(value, dict):
		chunks = []
		for v in value.values():
			chunks.extend(_flatten_text(v))
		return chunks
	if isinstance(value, (list, tuple, set)):
		chunks = []
		for item in value:
			chunks.extend(_flatten_text(item))
		return chunks
	return []


def _extract_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
	meta_fields = payload.get("metadata")
	if isinstance(meta_fields, dict):
		return meta_fields
	result: Dict[str, Any] = {}
	for key in ["title", "url", "category", "date", "version"]:
		if key in payload:
			result[key] = payload[key]
	return result


def build_dataframe(data_root: Path) -> pd.DataFrame:
	rows = []
	for json_path in sorted(data_root.glob("*/*.json")):
		if not json_path.is_file():
			continue
		try:
			json_obj = json.loads(json_path.read_text())
		except json.JSONDecodeError as exc:
			raise RuntimeError(f"Failed to parse {json_path}") from exc
		for payload in _iter_payloads(json_obj):
			text = _extract_text(payload)
			meta = _extract_meta(payload)
			last_modified = datetime.fromtimestamp(
				json_path.stat().st_mtime, tz=timezone.utc
			).isoformat()
			rows.append(
				{
					"source": str(json_path),
					"meta": meta,
					"text": text,
					"texts": text,
					"path": str(json_path),
					"page": 0,
					"last_modified_datetime": last_modified,
				}
			)
	if not rows:
		raise RuntimeError(f"No JSON files found under {data_root}")
	return pd.DataFrame(rows)


def main():
	parser = argparse.ArgumentParser(
		description="Convert JSON files under data/* into AutoRAG-compatible raw.parquet."
	)
	parser.add_argument(
		"--data-root",
		type=Path,
		default=Path("data"),
		help="Root directory that contains furiosa_* folders.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("raw.parquet"),
		help="Destination parquet file path.",
	)
	args = parser.parse_args()

	df = build_dataframe(args.data_root)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	df.to_parquet(args.output, index=False)
	print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
	main()
