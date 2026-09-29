# Controlled thermal matrix selection

Projection: `16.8 days × 46.27 measured-baseline genomes/min ÷ measured genomes/min`.
Acceptance gates: at least 20 minutes, peak at most 88 C, no throttle increment, zero worker failures, verified affinity, and 15–20 projected days.

| Profile | Control | Workers | Throughput (g/min) | Duration (min) | Peak C | Throttle | Failures | Affinity | Projection (d) | Accepted | Decision |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---|---|
| p8e10 | pmax4200_pl1_125w_pl2_160w | 18 | 42.96 | 22.3 | 84.0 | no | 0 | yes | 18.1 | yes | meets all measured gates |
| p8e10 | pmax4200_pl1_130w_pl2_160w | 18 | 41.87 | 22.9 | 86.0 | no | 0 | yes | 18.6 | yes | meets all measured gates |
| p6e12 | pmax4200_pl1_125w_pl2_160w | 18 | 40.00 | 24.0 | 85.0 | no | 0 | yes | 19.4 | yes | meets all measured gates |
| p8e10 | noturbo_pl1_125w_pl2_160w | 18 | 31.53 | 30.5 | 61.0 | no | 0 | yes | 24.7 | no | projection outside 15–20 d |
| p1e12 | none | 13 | 21.33 | 5.6 | 95.0 | no | 0 | yes | 36.4 | no | duration < 20 min; peak exceeds 88 C; projection outside 15–20 d |
| e10 | none | 10 | 19.25 | 12.5 | 88.0 | no | 0 | yes | 40.4 | no | duration < 20 min; projection outside 15–20 d |
| e12 | none | 12 | 18.03 | 6.7 | 91.0 | no | 0 | yes | 43.1 | no | duration < 20 min; peak exceeds 88 C; projection outside 15–20 d |
| e10 | none | 10 | 16.09 | 7.5 | 86.0 | no | 0 | yes | 48.3 | no | duration < 20 min; projection outside 15–20 d |
| e8 | none | 8 | 13.24 | 9.1 | 75.0 | no | 0 | yes | 58.7 | no | duration < 20 min; projection outside 15–20 d |
| p4e8 | none | 12 | — | 0.0 | 100.0 | yes | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; throttle counter incremented or unavailable; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p4e12 | none | 16 | — | 0.0 | 97.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p6e6 | none | 12 | — | 0.0 | 99.0 | yes | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; throttle counter incremented or unavailable; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p6e10 | none | 16 | — | 0.0 | 96.0 | yes | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; throttle counter incremented or unavailable; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p2e12 | none | 14 | — | 0.6 | 96.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p3e12 | none | 15 | — | 0.0 | 99.0 | yes | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; throttle counter incremented or unavailable; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p8e12 | cpuquota_1000 | 20 | — | 0.0 | 90.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p8e12 | cpuquota_1400 | 20 | — | 0.2 | 93.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p8e12 | cpuquota_1700 | 20 | — | 0.2 | 93.0 | yes | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; throttle counter incremented or unavailable; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p6e12 | pmax_4200mhz | 18 | — | 0.1 | 91.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p7e12 | pmax_4200mhz | 19 | — | 0.1 | 92.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p8e10 | pmax_4200mhz | 18 | — | 0.1 | 92.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p8e12 | pmax_4200mhz | 20 | — | 0.1 | 96.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p6e12 | pmax_3800mhz | 18 | — | 0.3 | 90.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p6e12 | pmax_3600mhz | 18 | — | 0.1 | 88.0 | no | — | — | — | no | safety abort; no completed throughput; duration < 20 min; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p7e12 | pmax4200_pl1_125w_pl2_160w | 19 | — | 22.0 | 89.0 | no | — | — | — | no | safety abort; no completed throughput; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p8e12 | pmax4200_pl1_125w_pl2_160w | 20 | — | 20.7 | 90.0 | no | — | — | — | no | safety abort; no completed throughput; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |
| p8e10 | pmax4200_pl1_135w_pl2_160w | 18 | — | 21.6 | 89.0 | no | — | — | — | no | safety abort; no completed throughput; peak exceeds 88 C; worker failure or unavailable; affinity unverified; projection outside 15–20 d |

## Recommendation

Select `p8e10` under `pmax4200_pl1_125w_pl2_160w`: 18.1 projected days with all gates satisfied.
