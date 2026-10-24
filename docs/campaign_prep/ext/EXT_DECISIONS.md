# Extension decisions (PRODUCTION)

Derived 2026-10-24T09:23:13+00:00 from preflight run `results/extension_preflight/ext-20261023` (commit `5e5a7a1e159387797997e8cea9ee8fea27985328`), calibrated on campaign `20260930T044002Z`. Rules: [docs/campaign_prep/ext/PREREGISTRATION_EXT.md](PREREGISTRATION_EXT.md).

| Task | Tier | ppo_timesteps | D6-ext branch | K | D16-ext branch | ratio single / max-of-2 | phase h (p5 / mean / p95) |
|---|---|---:|---|---:|---|---|---|
| ObstacleTraverser-v1 | hard | 600,000 | registered_knee | 1 | noise_rule | 3.009 / 1.865 | 117.2 / 130.3 / 143.3 |
| Climber-v0 | medium | 1,200,000 | fallback (disclosed floor) | 1 | noise_rule | 0.670 / 0.299 | 222.4 / 247.1 / 271.9 |
| CaveCrawler-v0 | medium | 500,000 | registered_knee | 1 | noise_rule | 1.077 / 0.761 | 98.5 / 109.5 / 120.4 |

**Projected campaign total:** 486.9 h mean (438.2–535.6 h, p5–p95) = 20.3 days. GUARD-ext threshold 360 h: **over_guard_proceed_with_later_completion**.

**Backtest:** the same method predicts the original UpStepper-v0 phase at 420.0 h against 388.9 h realised (+8.0%); the band half-width is max(10%, |backtest error|).

## Guard steps

- `none`: p95 total 943.4 h; design {"CaveCrawler-v0": {"ppo_seeds": 1, "ppo_timesteps": 500000}, "Climber-v0": {"ppo_seeds": 2, "ppo_timesteps": 1500000}, "ObstacleTraverser-v1": {"ppo_seeds": 1, "ppo_timesteps": 600000}}
- `a_K1_Climber-v0`: p95 total 603.5 h; design {"CaveCrawler-v0": {"ppo_seeds": 1, "ppo_timesteps": 500000}, "Climber-v0": {"ppo_seeds": 1, "ppo_timesteps": 1500000}, "ObstacleTraverser-v1": {"ppo_seeds": 1, "ppo_timesteps": 600000}}
- `b_T1.2M_Climber-v0`: p95 total 535.6 h; design {"CaveCrawler-v0": {"ppo_seeds": 1, "ppo_timesteps": 500000}, "Climber-v0": {"ppo_seeds": 1, "ppo_timesteps": 1200000}, "ObstacleTraverser-v1": {"ppo_seeds": 1, "ppo_timesteps": 600000}}

## Cost ratios (P-core-equivalent, same bodies, vs Walker-v0 at 1M)

- ObstacleTraverser-v1 at 600,000: 1.089 (bodies: direct:evolved, direct:random, nca:evolved, nca:random, rule_based:evolved, rule_based:random)
- Climber-v0 at 1,500,000: 2.582 (bodies: direct:evolved, direct:random, nca:evolved, nca:random, rule_based:evolved, rule_based:random)
- CaveCrawler-v0 at 500,000: 0.915 (bodies: direct:evolved, direct:random, nca:evolved, nca:random, rule_based:evolved, rule_based:random)
- UpStepper-v0 at 1,200,000: 1.213 (bodies: direct:evolved, direct:random, nca:evolved, nca:random, rule_based:evolved, rule_based:random)

Observed E/P training-time ratio in the preflight batch: 2.144

## Representative bodies

- `direct:evolved` hash `73be590c47e9` (evolved)
- `direct:random` hash `6d3f36dfb0b1` (random)
- `nca:evolved` hash `49463b126c87` (evolved)
- `nca:random` hash `1e0af7872ffd` (random)
- `rule_based:evolved` hash `02005b28a532` (evolved)
- `rule_based:random` hash `454c5469d8dc` (random)

## Preflight completeness

Recorded trainings: curves 18, noise (incl. references) 84.

Failed trainings: none.

Tasks flagged:
- none

## Raw rule outputs

```json
{
 "CaveCrawler-v0": {
  "d16_ext": {
   "branch": "noise_rule",
   "details": {
    "between_body_sd": 0.2512974887662569,
    "bodies_excluded": [],
    "bodies_used": [
     "direct:evolved",
     "direct:random",
     "nca:evolved",
     "nca:random",
     "rule_based:evolved",
     "rule_based:random"
    ],
    "bodies_with_max2_repeats": [
     "direct:evolved",
     "direct:random",
     "nca:evolved",
     "nca:random",
     "rule_based:evolved",
     "rule_based:random"
    ],
    "pooled_within_sd_max_of_2": 0.19129147869394922,
    "pooled_within_sd_single": 0.27062941893217485,
    "ratio_max_of_2": 0.7612152418756482,
    "ratio_single": 1.0769284653851017,
    "required_improvement": 0.5,
    "threshold": 0.33
   },
   "ppo_seeds": 1,
   "task": "CaveCrawler-v0"
  },
  "d6_ext": {
   "branch": "registered_knee",
   "clipped": "floor",
   "details": {
    "excluded_incomplete_bodies": [],
    "failed_trainings": [],
    "final_fitness": {
     "direct:evolved": 1.888051617981155,
     "direct:random": 1.9219676190401742,
     "nca:evolved": 0.7216950030851876,
     "nca:random": 1.7174122345570808,
     "rule_based:evolved": 1.8760127976144396,
     "rule_based:random": 1.9391989240871226
    },
    "knee": {
     "knee_found": true,
     "knee_timesteps": 200000.0,
     "knee_window_index": 1
    },
    "median_series": [
     [
      100000.0,
      1.6584817074998455
     ],
     [
      200000.0,
      1.8675105850954985
     ],
     [
      300000.0,
      1.8704416231025025
     ],
     [
      400000.0,
      1.8734287425893505
     ],
     [
      500000.0,
      1.8801151873981041
     ],
     [
      600000.0,
      1.8490082531659553
     ],
     [
      700000.0,
      1.892681096377279
     ],
     [
      800000.0,
      1.886548040057282
     ],
     [
      900000.0,
      1.8903570003119015
     ],
     [
      1000000.0,
      1.899952097823146
     ],
     [
      1100000.0,
      1.8729516589157829
     ],
     [
      1200000.0,
      1.905873732242064
     ],
     [
      1300000.0,
      1.9051599527446257
     ],
     [
      1400000.0,
      1.9076836709720482
     ],
     [
      1500000.0,
      1.9045874533620264
     ],
     [
      1600000.0,
      1.9077241508840004
     ],
     [
      1700000.0,
      1.904794262213012
     ],
     [
      1800000.0,
      1.9040641089262742
     ],
     [
      1900000.0,
      1.9066062530122032
     ],
     [
      2000000.0,
      1.8820322077977973
     ]
    ],
    "n_bodies_measured": 6,
    "n_bodies_used": 6,
    "raw_knee_checkpoint": 200000,
    "reduced_body_set": false,
    "rule_text": "The smallest budget at which the median marginal fitness improvement per additional 100k steps falls below 1% of the fitness achieved so far, sustained over three consecutive windows."
   },
   "disclosed_floor": false,
   "ppo_timesteps": 500000,
   "qualifying_bodies": [
    "direct:evolved",
    "direct:random",
    "nca:evolved",
    "nca:random",
    "rule_based:evolved",
    "rule_based:random"
   ],
   "task": "CaveCrawler-v0"
  }
 },
 "Climber-v0": {
  "d16_ext": {
   "branch": "noise_rule",
   "details": {
    "between_body_sd": 0.0853134275103861,
    "bodies_excluded": [],
    "bodies_used": [
     "direct:evolved",
     "direct:random",
     "nca:evolved",
     "nca:random",
     "rule_based:evolved",
     "rule_based:random"
    ],
    "bodies_with_max2_repeats": [
     "direct:evolved",
     "direct:random",
     "nca:evolved",
     "nca:random",
     "rule_based:evolved",
     "rule_based:random"
    ],
    "pooled_within_sd_max_of_2": 0.025469986433728407,
    "pooled_within_sd_single": 0.057165095215644446,
    "ratio_max_of_2": 0.29854604576316757,
    "ratio_single": 0.6700597653128534,
    "required_improvement": 0.5,
    "threshold": 0.33
   },
   "ppo_seeds": 2,
   "task": "Climber-v0"
  },
  "d6_ext": {
   "branch": "fallback",
   "clipped": "cap",
   "details": {
    "excluded_incomplete_bodies": [],
    "failed_trainings": [],
    "fallback_choice": 2000000,
    "fallback_table": [
     {
      "T": 1000000,
      "fraction_ok": false,
      "fractions_of_2M": {
       "direct:evolved": 0.5150870439940726,
       "direct:random": 1.4034347955057336,
       "nca:evolved": 6.337174161304222,
       "rule_based:random": 0.9950429429183154
      },
      "order": [
       "direct:random",
       "nca:evolved",
       "direct:evolved",
       "rule_based:random"
      ],
      "order_ok": false
     },
     {
      "T": 1200000,
      "fraction_ok": false,
      "fractions_of_2M": {
       "direct:evolved": 0.4261071290877256,
       "direct:random": 0.4356020480043662,
       "nca:evolved": 7.698885773151939,
       "rule_based:random": 0.9950429429183154
      },
      "order": [
       "nca:evolved",
       "direct:random",
       "direct:evolved",
       "rule_based:random"
      ],
      "order_ok": false
     },
     {
      "T": 1500000,
      "fraction_ok": false,
      "fractions_of_2M": {
       "direct:evolved": 0.7118518887719617,
       "direct:random": 0.9774390782195931,
       "nca:evolved": 8.876237771858134,
       "rule_based:random": 0.9950429429183154
      },
      "order": [
       "nca:evolved",
       "direct:random",
       "direct:evolved",
       "rule_based:random"
      ],
      "order_ok": false
     },
     {
      "T": 2000000,
      "fraction_ok": true,
      "fractions_of_2M": {
       "direct:evolved": 1.0,
       "direct:random": 1.0,
       "nca:evolved": 1.0,
       "rule_based:random": 1.0
      },
      "order": [
       "direct:random",
       "direct:evolved",
       "nca:evolved",
       "rule_based:random"
      ],
      "order_ok": true
     }
    ],
    "final_fitness": {
     "direct:evolved": 0.06023985801584786,
     "direct:random": 0.18889574767797512,
     "nca:evolved": 0.02368909334836644,
     "nca:random": -0.01091973473012331,
     "rule_based:evolved": 0.012690255233337666,
     "rule_based:random": 0.020372709510525056
    },
    "final_order": [
     "direct:random",
     "direct:evolved",
     "nca:evolved",
     "rule_based:random"
    ],
    "knee": {
     "knee_found": false,
     "knee_timesteps": null,
     "knee_window_index": null
    },
    "median_series": [
     [
      100000.0,
      0.005345469286549331
     ],
     [
      200000.0,
      0.07936155685689528
     ],
     [
      300000.0,
      0.11476830981511998
     ],
     [
      400000.0,
      0.021705788099937173
     ],
     [
      500000.0,
      -0.00261099570198553
     ],
     [
      600000.0,
      0.013618097514665806
     ],
     [
      700000.0,
      0.12277478187206725
     ],
     [
      800000.0,
      0.08496045587330059
     ],
     [
      900000.0,
      0.11500181065450565
     ],
     [
      1000000.0,
      0.09057534033399861
     ],
     [
      1100000.0,
      0.09648825492695323
     ],
     [
      1200000.0,
      0.05397600375181355
     ],
     [
      1300000.0,
      0.058677600152094556
     ],
     [
      1400000.0,
      0.03088973337246248
     ],
     [
      1500000.0,
      0.11375797109894847
     ],
     [
      1600000.0,
      0.038672038033767286
     ],
     [
      1700000.0,
      0.038470026613070196
     ],
     [
      1800000.0,
      0.1339241621998326
     ],
     [
      1900000.0,
      0.09050066367967702
     ],
     [
      2000000.0,
      0.04196447568210715
     ]
    ],
    "n_bodies_measured": 6,
    "n_bodies_used": 6,
    "reduced_body_set": false,
    "rule_text": "The smallest budget at which the median marginal fitness improvement per additional 100k steps falls below 1% of the fitness achieved so far, sustained over three consecutive windows."
   },
   "disclosed_floor": true,
   "ppo_timesteps": 1500000,
   "qualifying_bodies": [
    "direct:evolved",
    "direct:random",
    "nca:evolved",
    "rule_based:random"
   ],
   "task": "Climber-v0"
  }
 },
 "ObstacleTraverser-v1": {
  "d16_ext": {
   "branch": "noise_rule",
   "details": {
    "between_body_sd": 0.11403446564199099,
    "bodies_excluded": [],
    "bodies_used": [
     "direct:evolved",
     "direct:random",
     "nca:evolved",
     "nca:random",
     "rule_based:evolved",
     "rule_based:random"
    ],
    "bodies_with_max2_repeats": [
     "direct:evolved",
     "direct:random",
     "nca:evolved",
     "nca:random",
     "rule_based:evolved",
     "rule_based:random"
    ],
    "pooled_within_sd_max_of_2": 0.2127097378366064,
    "pooled_within_sd_single": 0.34318008370303194,
    "ratio_max_of_2": 1.8653109534831736,
    "ratio_single": 3.0094417663203616,
    "required_improvement": 0.5,
    "threshold": 0.33
   },
   "ppo_seeds": 1,
   "task": "ObstacleTraverser-v1"
  },
  "d6_ext": {
   "branch": "registered_knee",
   "clipped": null,
   "details": {
    "excluded_incomplete_bodies": [],
    "failed_trainings": [],
    "final_fitness": {
     "direct:evolved": 1.087664987601272,
     "direct:random": 1.150332170987788,
     "nca:evolved": 1.109050133452314,
     "nca:random": 1.1254413853709586,
     "rule_based:evolved": 3.099006798841856,
     "rule_based:random": 1.2559238131141512
    },
    "knee": {
     "knee_found": true,
     "knee_timesteps": 600000.0,
     "knee_window_index": 5
    },
    "median_series": [
     [
      100000.0,
      0.2928647831782095
     ],
     [
      200000.0,
      0.46101303534516447
     ],
     [
      300000.0,
      0.7987337553416367
     ],
     [
      400000.0,
      0.7847727244285698
     ],
     [
      500000.0,
      1.0729999580985283
     ],
     [
      600000.0,
      1.1329577705568505
     ],
     [
      700000.0,
      1.1075398925021904
     ],
     [
      800000.0,
      1.1178090352594765
     ],
     [
      900000.0,
      1.1094259689347439
     ],
     [
      1000000.0,
      1.115021302864172
     ],
     [
      1100000.0,
      1.1357736527197462
     ],
     [
      1200000.0,
      1.1624047523032495
     ],
     [
      1300000.0,
      1.1465269836743959
     ],
     [
      1400000.0,
      1.161630626388317
     ],
     [
      1500000.0,
      1.100449985104008
     ],
     [
      1600000.0,
      1.1528095439995276
     ],
     [
      1700000.0,
      1.148295700043401
     ],
     [
      1800000.0,
      1.130681012578524
     ],
     [
      1900000.0,
      1.155143010199641
     ],
     [
      2000000.0,
      1.1378867781793733
     ]
    ],
    "n_bodies_measured": 6,
    "n_bodies_used": 6,
    "raw_knee_checkpoint": 600000,
    "reduced_body_set": false,
    "rule_text": "The smallest budget at which the median marginal fitness improvement per additional 100k steps falls below 1% of the fitness achieved so far, sustained over three consecutive windows."
   },
   "disclosed_floor": false,
   "ppo_timesteps": 600000,
   "qualifying_bodies": [
    "direct:evolved",
    "direct:random",
    "nca:evolved",
    "nca:random",
    "rule_based:evolved",
    "rule_based:random"
   ],
   "task": "ObstacleTraverser-v1"
  }
 }
}
```
