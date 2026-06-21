"""Single source of truth for an optimization run's output-directory layout.

A run writes two kinds of artifacts under a base directory: headline deliverables
at the top level and secondary/cost/internal files under ``details/``. The
optimizer writes result artifacts relative to ``output_dir`` and exam-cache
artifacts relative to ``cache_dir`` (equal in a plain ``optimize`` run, split when
a driver passes ``output_dir_override``), so ``RunLayout`` is constructed from
whichever base applies and the caller picks the base. The benchmark repo imports
these names so the two repos cannot drift.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

DETAILS_DIRNAME = "details"
DEBUG_DIRNAME = "debug"
FRONTIER_DIRNAME = "frontier"

SUMMARY_FILE = "optimization_summary.md"
RECOMMENDED_FILE = "recommended.yaml"
RUN_LOG_FILE = "run.log"
EXAM_FILE = "exam.json"

HISTORY_FILE = "history.jsonl"
COST_BREAKDOWN_FILE = "cost_breakdown.json"
TRIAL_COST_LEDGER_FILE = "trial_cost_ledger.jsonl"
CANDIDATES_FILE = "candidates.json"
EXAM_COST_FILE = "exam_cost.json"


class RunLayout(BaseModel):
    """Resolves every output path under ``base``. Base-agnostic by design."""

    model_config = ConfigDict(frozen=True)

    base: Path

    @property
    def details(self) -> Path:
        return self.base / DETAILS_DIRNAME

    @property
    def debug(self) -> Path:
        return self.details / DEBUG_DIRNAME

    @property
    def frontier_dir(self) -> Path:
        return self.base / FRONTIER_DIRNAME

    @property
    def summary(self) -> Path:
        return self.base / SUMMARY_FILE

    @property
    def recommended(self) -> Path:
        return self.base / RECOMMENDED_FILE

    @property
    def run_log(self) -> Path:
        return self.base / RUN_LOG_FILE

    @property
    def exam(self) -> Path:
        return self.base / EXAM_FILE

    @property
    def history(self) -> Path:
        return self.details / HISTORY_FILE

    @property
    def cost_breakdown(self) -> Path:
        return self.details / COST_BREAKDOWN_FILE

    @property
    def trial_cost_ledger(self) -> Path:
        return self.details / TRIAL_COST_LEDGER_FILE

    @property
    def candidates(self) -> Path:
        return self.details / CANDIDATES_FILE

    @property
    def exam_cost(self) -> Path:
        return self.details / EXAM_COST_FILE

    def ensure_details(self) -> Path:
        self.details.mkdir(parents=True, exist_ok=True)
        return self.details

    def ensure_debug(self) -> Path:
        self.debug.mkdir(parents=True, exist_ok=True)
        return self.debug
