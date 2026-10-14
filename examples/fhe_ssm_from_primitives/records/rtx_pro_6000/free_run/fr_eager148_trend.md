
## fr_eager148.json: mode eager, 148 ticks, 64 conversations, policy {'policy': 'norepeat', 'penalty': 1.3, 'window': 64, 'ngram': 4, 'ban': [0]}

| ticks | mean logits relErrRms (aggregate) | median over the block's ticks | mean worst conversation | top-1 flips | pick flips | generating conv-ticks | on-record agree | median s/tick | runtime boots/tick | free GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-9 | 0.004494 | 0.004337 | 0.025064 | 1 | 1 | 111/640 | 111/111 | 74.39 | [1968] | [60.9, 62.9] |
| 10-19 | 0.004995 | 0.004780 | 0.033818 | 3 | 1 | 312/640 | 302/304 | 74.26 | [1968] | [60.9] |
| 20-29 | 0.004604 | 0.004693 | 0.026508 | 6 | 3 | 320/640 | 270/275 | 74.38 | [1968] | [60.9] |
| 30-39 | 0.004902 | 0.004878 | 0.033076 | 5 | 5 | 353/640 | 261/264 | 74.31 | [1968] | [60.9] |
| 40-49 | 0.004995 | 0.004904 | 0.023986 | 5 | 6 | 462/640 | 349/351 | 74.25 | [1968] | [60.9] |
| 50-59 | 0.004950 | 0.004888 | 0.045717 | 5 | 5 | 480/640 | 355/359 | 74.38 | [1968] | [60.9] |
| 60-69 | 0.005089 | 0.005077 | 0.036252 | 5 | 4 | 500/640 | 335/337 | 74.38 | [1968] | [60.9] |
| 70-79 | 0.004555 | 0.004482 | 0.023772 | 8 | 8 | 591/640 | 369/376 | 74.34 | [1968] | [60.9] |
| 80-89 | 0.004805 | 0.004781 | 0.026377 | 4 | 6 | 640/640 | 383/386 | 74.39 | [1968] | [60.9] |
| 90-99 | 0.005012 | 0.004837 | 0.032263 | 9 | 7 | 640/640 | 338/342 | 74.43 | [1968] | [60.9] |
| 100-109 | 0.005070 | 0.004799 | 0.028929 | 3 | 6 | 640/640 | 296/301 | 74.45 | [1968] | [60.9] |
| 110-119 | 0.004780 | 0.004705 | 0.029349 | 5 | 3 | 640/640 | 270/270 | 74.24 | [1968] | [60.9] |
| 120-129 | 0.004520 | 0.004595 | 0.025713 | 5 | 9 | 640/640 | 259/261 | 74.22 | [1968] | [60.9] |
| 130-139 | 0.005178 | 0.005065 | 0.028322 | 9 | 6 | 640/640 | 235/237 | 74.28 | [1968] | [60.9] |
| 140-147 | 0.005070 | 0.005029 | 0.027009 | 3 | 3 | 512/512 | 184/184 | 74.14 | [1968] | [60.9] |

- raw top-1 agreement 9396/9472 = 99.198 %; pick agreement 9399/9472 = 99.229 %
- seconds per tick: first 85.37, median 74.34, min 74.06, max 85.37
- least-squares power law through 14 full blocks: level = 0.004527 * tick^0.017 (description of these records only)
- block level first/last: 0.004494 / 0.005178 (ratio 1.15)
- conversations that left the record: 41 of 64: conv 2 at pos 31, conv 3 at pos 23, conv 5 at pos 75, conv 6 at pos 70, conv 7 at pos 15, conv 9 at pos 45, conv 13 at pos 87, conv 14 at pos 110, conv 17 at pos 38, conv 18 at pos 90, conv 20 at pos 132, conv 21 at pos 94, conv 22 at pos 78, conv 26 at pos 123, conv 27 at pos 105, conv 28 at pos 106, conv 29 at pos 135, conv 30 at pos 25, conv 32 at pos 104, conv 36 at pos 76, conv 37 at pos 92, conv 38 at pos 32, conv 39 at pos 27, conv 40 at pos 106, conv 41 at pos 100, conv 43 at pos 17, conv 44 at pos 77, conv 45 at pos 60, conv 46 at pos 60, conv 47 at pos 59, conv 49 at pos 128, conv 51 at pos 72, conv 52 at pos 89, conv 53 at pos 44, conv 54 at pos 72, conv 55 at pos 27, conv 58 at pos 60, conv 59 at pos 67, conv 60 at pos 96, conv 62 at pos 75, conv 63 at pos 23
  - conv 2 at pos 31: own flip (closed form predicts None); pick at pos 30 was 9666 (top-1 45090); pick agreement with the closed form at that tick 0.984375
  - conv 3 at pos 23: own flip (closed form predicts None); pick at pos 22 was 2716 (top-1 2716); pick agreement with the closed form at that tick 0.96875
  - conv 5 at pos 75: own flip (closed form predicts None); pick at pos 74 was 840 (top-1 840); pick agreement with the closed form at that tick 0.96875
  - conv 6 at pos 70: own flip (closed form predicts None); pick at pos 69 was 13 (top-1 13); pick agreement with the closed form at that tick 0.984375
  - conv 7 at pos 15: own flip (closed form predicts None); pick at pos 14 was 1379 (top-1 1379); pick agreement with the closed form at that tick 0.984375
  - conv 9 at pos 45: own flip (closed form predicts None); pick at pos 44 was 352 (top-1 253); pick agreement with the closed form at that tick 0.984375
  - conv 13 at pos 87: own flip (closed form predicts None); pick at pos 86 was 4916 (top-1 310); pick agreement with the closed form at that tick 0.984375
  - conv 14 at pos 110: own flip (closed form predicts 43); pick at pos 109 was 534 (top-1 25818); pick agreement with the closed form at that tick 0.984375
  - conv 17 at pos 38: own flip (closed form predicts 146); pick at pos 37 was 1984 (top-1 7741); pick agreement with the closed form at that tick 0.96875
  - conv 18 at pos 90: own flip (closed form predicts None); pick at pos 89 was 271 (top-1 581); pick agreement with the closed form at that tick 0.96875
  - conv 20 at pos 132: own flip (closed form predicts None); pick at pos 131 was 4444 (top-1 5202); pick agreement with the closed form at that tick 0.984375
  - conv 21 at pos 94: predicted; pick at pos 93 was 322 (top-1 367); pick agreement with the closed form at that tick 1.0
  - conv 22 at pos 78: own flip (closed form predicts 86); pick at pos 77 was 7899 (top-1 7899); pick agreement with the closed form at that tick 0.984375
  - conv 26 at pos 123: own flip (closed form predicts None); pick at pos 122 was 8588 (top-1 7952); pick agreement with the closed form at that tick 0.984375
  - conv 27 at pos 105: own flip (closed form predicts None); pick at pos 104 was 763 (top-1 7590); pick agreement with the closed form at that tick 0.984375
  - conv 28 at pos 106: own flip (closed form predicts None); pick at pos 105 was 969 (top-1 13); pick agreement with the closed form at that tick 0.96875
  - conv 29 at pos 135: own flip (closed form predicts None); pick at pos 134 was 534 (top-1 534); pick agreement with the closed form at that tick 0.984375
  - conv 30 at pos 25: own flip (closed form predicts None); pick at pos 24 was 2544 (top-1 310); pick agreement with the closed form at that tick 0.984375
  - conv 32 at pos 104: own flip (closed form predicts None); pick at pos 103 was 43 (top-1 43); pick agreement with the closed form at that tick 0.984375
  - conv 36 at pos 76: own flip (closed form predicts 77); pick at pos 75 was 327 (top-1 1335); pick agreement with the closed form at that tick 0.984375
  - conv 37 at pos 92: own flip (closed form predicts None); pick at pos 91 was 617 (top-1 281); pick agreement with the closed form at that tick 0.96875
  - conv 38 at pos 32: own flip (closed form predicts None); pick at pos 31 was 715 (top-1 715); pick agreement with the closed form at that tick 0.984375
  - conv 39 at pos 27: predicted; pick at pos 26 was 2264 (top-1 2264); pick agreement with the closed form at that tick 1.0
  - conv 40 at pos 106: own flip (closed form predicts None); pick at pos 105 was 44091 (top-1 253); pick agreement with the closed form at that tick 0.96875
  - conv 41 at pos 100: own flip (closed form predicts None); pick at pos 99 was 352 (top-1 310); pick agreement with the closed form at that tick 0.96875
  - conv 43 at pos 17: predicted; pick at pos 16 was 1263 (top-1 2798); pick agreement with the closed form at that tick 1.0
  - conv 44 at pos 77: own flip (closed form predicts None); pick at pos 76 was 1824 (top-1 1824); pick agreement with the closed form at that tick 0.984375
  - conv 45 at pos 60: own flip (closed form predicts 78); pick at pos 59 was 1481 (top-1 14910); pick agreement with the closed form at that tick 0.9375
  - conv 46 at pos 60: own flip (closed form predicts None); pick at pos 59 was 6089 (top-1 187); pick agreement with the closed form at that tick 0.9375
  - conv 47 at pos 59: predicted; pick at pos 58 was 8859 (top-1 4177); pick agreement with the closed form at that tick 0.984375
  - conv 49 at pos 128: own flip (closed form predicts 96); pick at pos 127 was 253 (top-1 253); pick agreement with the closed form at that tick 0.984375
  - conv 51 at pos 72: predicted; pick at pos 71 was 534 (top-1 253); pick agreement with the closed form at that tick 0.984375
  - conv 52 at pos 89: own flip (closed form predicts None); pick at pos 88 was 7985 (top-1 281); pick agreement with the closed form at that tick 0.984375
  - conv 53 at pos 44: own flip (closed form predicts 123); pick at pos 43 was 1824 (top-1 8281); pick agreement with the closed form at that tick 0.96875
  - conv 54 at pos 72: own flip (closed form predicts None); pick at pos 71 was 840 (top-1 840); pick agreement with the closed form at that tick 0.984375
  - conv 55 at pos 27: predicted; pick at pos 26 was 840 (top-1 310); pick agreement with the closed form at that tick 1.0
  - conv 58 at pos 60: own flip (closed form predicts 143); pick at pos 59 was 1211 (top-1 1211); pick agreement with the closed form at that tick 0.9375
  - conv 59 at pos 67: own flip (closed form predicts None); pick at pos 66 was 281 (top-1 281); pick agreement with the closed form at that tick 0.984375
  - conv 60 at pos 96: own flip (closed form predicts None); pick at pos 95 was 43390 (top-1 4669); pick agreement with the closed form at that tick 0.96875
  - conv 62 at pos 75: own flip (closed form predicts None); pick at pos 74 was 310 (top-1 310); pick agreement with the closed form at that tick 0.96875
  - conv 63 at pos 23: predicted; pick at pos 22 was 2739 (top-1 2341); pick agreement with the closed form at that tick 0.96875
- predicted departures due by pos 147: 16; of them made: 7, pre-empted by an earlier own flip: 6, not made because this run flipped the same near-tie as the published run (it picked the record's token, the closed form's second choice, and stays on the record): 1 [(84, 23)]; the same, but the conversation left later by an own flip: 2 [(43, 14, 110), (96, 49, 128)]
