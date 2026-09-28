"""Resume helpers; journal hashes make proposal drift a hard error."""
from __future__ import annotations

from .checkpoint import latest_checkpoint
from .journal import ResultJournal
from .scheduler import genome_hash


def assert_reproposal_matches_journal(run, encoder, checkpoint_dir, journal_path, run_key):
    """Load a generation boundary and verify any journaled pending genome.

    A mismatch means code/config/RNG state drifted; continuing would create a
    superficially complete but scientifically invalid campaign, so fail loud.
    """
    state = latest_checkpoint(checkpoint_dir)
    if state is None:
        return None
    journal = ResultJournal(journal_path).indexed()
    candidates = run.propose()
    for candidate in candidates:
        entries = [record for (key, index, _k), record in journal.items()
                   if key == run_key and index == candidate.eval_index]
        if entries and any(record["genome_hash"] != genome_hash(encoder, candidate.genome) for record in entries):
            raise RuntimeError(f"genome-hash mismatch while resuming {run_key} eval_index={candidate.eval_index}")
    return candidates
