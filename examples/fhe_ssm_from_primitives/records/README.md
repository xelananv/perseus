# Records

Measurements of this port, with the raw outputs they are read from. Every number in these READMEs names the file
(and line or field) it comes from; nothing is quoted from memory. Timings compare only within one build on one card.
Local working-directory paths in the Mac-side logs are shortened to `<work>/`, `<perseus>/`, `<home>/` and `<tmp>/`;
nothing else in any log is changed.

| folder | what | when, where |
|---|---|---|
| [`a6000_capture/`](a6000_capture/) | the capture with the host decrypt vs the device magnitude probe: time, agreement node by node, the plans made from each | 12 Oct 2026, one RTX A6000, branch `fhe-ssm-port` |
| [`dev_planner/`](dev_planner/) | upstream dev's planner on this model's capture, before and after the re-cut fix; the shipped-plan battery on both | 13 Oct 2026, CPU (Mac), branch `fhe-ssm-port-dev` |
| [`rtx_pro_6000/`](rtx_pro_6000/) | the port on upstream dev on Blackwell: the verification chain, the device probes, the plan and its equality across cards, two 148-tick free-running runs (planned and eager), a same-card eager/planned timing pair | 13 Oct 2026, 2× RTX PRO 6000 Blackwell, branch `fhe-ssm-port-dev` |

The graph captures the plans were made from are 45–47 MB each and are not in git; their archives and sha256 are listed
in the folders that use them.
