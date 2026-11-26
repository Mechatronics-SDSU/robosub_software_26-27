"""Run the ESKF against a recorded sensor CSV."""

from __future__ import annotations

import argparse

from modules.navigation.replay import read_samples, run_replay, summarize_replay, write_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Recorded sensor CSV")
    parser.add_argument("output", help="Output estimated-state CSV")
    args = parser.parse_args()
    samples = list(read_samples(args.input))
    states = run_replay(samples)
    write_states(args.output, states)
    print(f"Replayed {len(states)} samples to {args.output}")
    print(summarize_replay(samples, states).as_json())


if __name__ == "__main__":
    main()
