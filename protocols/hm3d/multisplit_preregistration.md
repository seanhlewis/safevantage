# HM3D multi-split evaluation protocol

Frozen on 2026-08-23 before computing the results reported here.

The evaluation tests four questions with fixed score streams:

1. Is the four-scene calibration split unusually favorable?
2. Does SafeVantage still win when every baseline is forced to the same coverage?
3. Does the result hold over an entire calibration-selected risk--coverage curve,
   rather than at one operating point?
4. Are the conclusions stable under scene-clustered resampling?

The 36 HM3D-Sem scenes are sorted by SHA-256 and partitioned into nine disjoint
four-scene calibration folds. For each fold, thresholds are selected using only
its four calibration scenes and evaluated on the other 32 scenes. Every scene is
therefore calibration exactly once and evaluation eight times. SafeVantage uses
its support count; caption-RAG uses its frozen BM25 score; ConceptGraphs and
VLMaps use their frozen raw support scores. All arms use the same 160 input views.

For matched coverage, SafeVantage first selects its minimum-risk two-threshold
operating point on calibration. Each baseline then selects, on calibration only,
the threshold pair nearest to SafeVantage's calibration coverage, breaking ties
by lower risk and then lower false-positive rate. For the AURC diagnostic, each
method selects one threshold pair per declared coverage target using calibration
only; the resulting ten test risks are integrated with the trapezoid rule.

The analysis is robustness evidence, not a fresh benchmark. No result may be
described as an unused-scene confirmation. Scene-bootstrap intervals resample
whole scenes after averaging repeated out-of-calibration predictions for each
case. All reversals and all failed gates must be reported.

## Explicit negative-closure audit

Before computing the result, we additionally freeze a bounded post-hoc audit on
the immutable 18-scene ScanNet four-state episodes. A negative certificate may
use only observed opportunity coverage and existing evidence scores. On the six
development scenes, select the rule with maximum observed-absent recall subject
to at most 5% false negative commitments on all non-absent states. Evaluate once
on the twelve locked scenes and compare with a score-only low-confidence rule
selected under the identical constraint. The certificate passes only if test
false-negative commitment is at most 10%, observed-absent recall is at least
80%, and it improves recall over score-only by at least 20 points.

This is an observability-closure unit test: the episode constructor deliberately
gives observed-absent prefixes broader sampled coverage than unobserved prefixes.
It cannot be called proof that arbitrary real trajectories exhaust a scene.
