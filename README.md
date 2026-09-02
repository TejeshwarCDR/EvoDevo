# EvoDevo: Direct vs. Developmental (NCA) Encodings in EvoGym

**Research question:** Under an identical evaluation pipeline, fixed 5x5 grid, and equal evaluation budget, does an NCA developmental encoding differ from a direct encoding in final task fitness, search efficiency, and morphological properties — and in which direction?

This project makes no assumption that the developmental encoding wins. The architecture is deliberately symmetric so that the comparison, not the encoding, is the deliverable.

## One pipeline, two decoders

Both encodings share every stage of the pipeline except the decoder itself:

```
genome -> Encoder.decode(genome) -> body (5x5 int array)
       -> morphology validator                [shared]
       -> get_full_connectivity()             [shared, EvoGym]
       -> gym.make('Walker-v0', ...)          [shared, EvoGym]
       -> PPO training (fixed budget)          [shared]
       -> evaluation reward -> fitness         [shared]
       -> EA selection on the genome           [shared]
```

Everything below the encoder box is written once, so that any observed difference in results comes from the representation, not from asymmetries in the pipeline.

See `docs/decisions.md` for every implementation decision not settled by prior project constraints, and `evodevo-implementation-roadmap.md` for the full build plan.

## Status

Repository skeleton only. No encoder, evolutionary, or PPO logic has been implemented yet.
