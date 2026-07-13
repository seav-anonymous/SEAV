# OrdSense

OrdSense is the ordering-sensitivity diagnostic dataset introduced with SEAV.
It contains 137 newline-delimited JSON records. Each record represents one
procedure under three orderings:

- `original_response`: the original procedure ordering.
- `alt_correct_response`: a different topological ordering that preserves all
  inferred dependencies.
- `wrong_order_response`: an ordering that violates at least one inferred
  dependency.

The three variants use the same step content and differ only in ordering. See
Appendix B.2 of the paper for the full construction procedure, including
moderation, repeated dependency labeling, consistency filtering, edge
cleaning, and variant generation.

## File

`order_variants_with_responses.jsonl` contains all 137 records used in the
paper. It is UTF-8 JSONL with one object per line.

## Source and terms

OrdSense is derived from the publicly released How2Bench and How2Train data
from *How2Everything: Mining the Web for How-To Procedures to Evaluate and
Improve LLMs*:

- Project: https://github.com/lilakk/how2everything
- Data: https://huggingface.co/collections/how2everything/how2everything-data
- Paper: https://arxiv.org/abs/2602.08808

The procedural text originates from the How2Everything release and remains
subject to its source terms. This dataset is not covered by the repository's
MIT software license. Users are responsible for complying with applicable
source-data and content terms.

## Safety notice

This research dataset contains harmful or illicit procedural intents and
responses. It is intended for safety evaluation and research, not operational
use.
