
## fr_planned148.json: mode planned, 148 ticks, 64 conversations, policy {'policy': 'norepeat', 'penalty': 1.3, 'window': 64, 'ngram': 4, 'ban': [0]}
- plan runtime switches: {'plan': {'FIDESLIB_BTS_REAL': '1'}, 'effective': {'FIDESLIB_BTS_REAL': '1'}}

| ticks | mean logits relErrRms (aggregate) | median over the block's ticks | mean worst conversation | top-1 flips | pick flips | generating conv-ticks | on-record agree | median s/tick | runtime boots/tick | free GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-9 | 0.000534 | 0.000487 | 0.004586 | 0 | 0 | 111/640 | 111/111 | 46.41 | [953] | [59.9, 62.9] |
| 10-19 | 0.000672 | 0.000630 | 0.005878 | 0 | 0 | 312/640 | 308/309 | 46.31 | [953] | [59.9] |
| 20-29 | 0.001120 | 0.000729 | 0.013160 | 0 | 1 | 320/640 | 302/304 | 46.37 | [953] | [59.9] |
| 30-39 | 0.000788 | 0.000595 | 0.008574 | 0 | 0 | 353/640 | 323/323 | 46.30 | [953] | [59.9] |
| 40-49 | 0.001003 | 0.000927 | 0.007292 | 0 | 0 | 462/640 | 424/425 | 46.22 | [953] | [59.9] |
| 50-59 | 0.000744 | 0.000718 | 0.006247 | 0 | 0 | 480/640 | 438/439 | 46.16 | [953] | [59.9] |
| 60-69 | 0.001156 | 0.000912 | 0.013144 | 0 | 0 | 500/640 | 450/450 | 46.27 | [953] | [59.9] |
| 70-79 | 0.001080 | 0.000948 | 0.010917 | 0 | 0 | 591/640 | 525/528 | 46.31 | [953] | [59.9] |
| 80-89 | 0.000942 | 0.000738 | 0.010515 | 1 | 1 | 640/640 | 547/550 | 46.27 | [953] | [59.9] |
| 90-99 | 0.001105 | 0.001001 | 0.007004 | 1 | 0 | 640/640 | 518/520 | 46.27 | [953] | [59.9] |
| 100-109 | 0.000752 | 0.000723 | 0.005237 | 0 | 2 | 640/640 | 502/504 | 46.42 | [953] | [59.9] |
| 110-119 | 0.000764 | 0.000740 | 0.005390 | 0 | 0 | 640/640 | 490/490 | 46.47 | [953] | [59.9] |
| 120-129 | 0.000945 | 0.000812 | 0.007161 | 1 | 1 | 640/640 | 482/483 | 46.34 | [953] | [59.9] |
| 130-139 | 0.001453 | 0.001139 | 0.013495 | 0 | 0 | 640/640 | 480/480 | 46.37 | [953] | [59.9] |
| 140-147 | 0.001447 | 0.001306 | 0.024419 | 1 | 2 | 512/512 | 378/379 | 46.42 | [953] | [59.9] |

- raw top-1 agreement 9468/9472 = 99.958 %; pick agreement 9465/9472 = 99.926 %
- seconds per tick: first 58.72, median 46.34, min 45.84, max 58.72
- least-squares power law through 14 full blocks: level = 0.000455 * tick^0.173 (description of these records only)
- block level first/last: 0.000534 / 0.001453 (ratio 2.72)
- conversations that left the record: 17 of 64: conv 0 at pos 108, conv 9 at pos 90, conv 14 at pos 43, conv 21 at pos 94, conv 22 at pos 86, conv 23 at pos 84, conv 36 at pos 77, conv 39 at pos 27, conv 40 at pos 106, conv 43 at pos 17, conv 45 at pos 78, conv 47 at pos 59, conv 49 at pos 96, conv 51 at pos 72, conv 53 at pos 123, conv 55 at pos 27, conv 58 at pos 143
  - conv 0 at pos 108: own flip (closed form predicts None); pick at pos 107 was 6311 (top-1 247); pick agreement with the closed form at that tick 0.984375
  - conv 9 at pos 90: own flip (closed form predicts None); pick at pos 89 was 3249 (top-1 3249); pick agreement with the closed form at that tick 0.984375
  - conv 14 at pos 43: predicted; pick at pos 42 was 32632 (top-1 187); pick agreement with the closed form at that tick 1.0
  - conv 21 at pos 94: predicted; pick at pos 93 was 322 (top-1 367); pick agreement with the closed form at that tick 1.0
  - conv 22 at pos 86: predicted; pick at pos 85 was 247 (top-1 247); pick agreement with the closed form at that tick 1.0
  - conv 23 at pos 84: predicted; pick at pos 83 was 7127 (top-1 7652); pick agreement with the closed form at that tick 1.0
  - conv 36 at pos 77: predicted; pick at pos 76 was 13828 (top-1 13); pick agreement with the closed form at that tick 1.0
  - conv 39 at pos 27: predicted; pick at pos 26 was 2264 (top-1 2264); pick agreement with the closed form at that tick 1.0
  - conv 40 at pos 106: own flip (closed form predicts None); pick at pos 105 was 44091 (top-1 253); pick agreement with the closed form at that tick 0.984375
  - conv 43 at pos 17: predicted; pick at pos 16 was 1263 (top-1 2798); pick agreement with the closed form at that tick 1.0
  - conv 45 at pos 78: predicted; pick at pos 77 was 476 (top-1 671); pick agreement with the closed form at that tick 1.0
  - conv 47 at pos 59: predicted; pick at pos 58 was 8859 (top-1 4177); pick agreement with the closed form at that tick 1.0
  - conv 49 at pos 96: predicted; pick at pos 95 was 1390 (top-1 320); pick agreement with the closed form at that tick 1.0
  - conv 51 at pos 72: predicted; pick at pos 71 was 534 (top-1 253); pick agreement with the closed form at that tick 1.0
  - conv 53 at pos 123: predicted; pick at pos 122 was 1048 (top-1 8281); pick agreement with the closed form at that tick 0.984375
  - conv 55 at pos 27: predicted; pick at pos 26 was 840 (top-1 310); pick agreement with the closed form at that tick 1.0
  - conv 58 at pos 143: predicted; pick at pos 142 was 19607 (top-1 19607); pick agreement with the closed form at that tick 1.0
- predicted departures due by pos 147: 16; of them made: 14, pre-empted by an earlier own flip: 0, not made because this run flipped the same near-tie as the published run (it picked the record's token, the closed form's second choice, and stays on the record): 2 [(146, 17), (23, 63)]; the same, but the conversation left later by an own flip: 0
