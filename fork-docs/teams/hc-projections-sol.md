# HC projection implementation handoff

The independent down, up, and inject switches are implemented in the validation worktree. Use `start-dialogue-measurement.ps1 -Mode speed -HcAll` for the first full-model run after rebuilding. The previous `-HcUpPair` run used a different reduction layout and remains a separate experiment.

The CUDA DLL and graph fixture are under the Git-external `hc-scale-check/projections` validation directory. Use the `final-*` series. Earlier smoke files precede the final up row grouping. The user owns the full build and model run. [Implementation and fixture record](../research/hc-projections.md)
