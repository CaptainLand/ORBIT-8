# ORBIT-8 Trans-1 Dynamic v2

## Training

- Source charts: prepared-v2 official charts, levels 12-15.
- Dynamic anchors: prepared-v3 regular, dense, long-object, double, and silence pools.
- Crops: 8, 12, and 16 measures.
- Augmentation: normal, horizontal mirror, vertical mirror, and half turn.
- Samples per epoch: 16,208.
- Epochs: 6.
- Parameters: 2,958,842.
- Warm start: Trans-1 hybrid v1.

## Best checkpoint

Epoch 6 was selected with a validation composite score of 0.617883.

| Metric | Dynamic v2 | Previous Trans-1 best |
| --- | ---: | ---: |
| Lane accuracy | 43.46% | 40.76% |
| Operator accuracy | 48.87% | 49.17% |
| Interaction recall | 77.66% | 80.44% |
| Sweep recall | 83.71% | 79.54% |
| Composite score | 0.6179 | about 0.588 |

The dynamic model improves general lane placement and sweep recognition. Its
interaction recall is lower than the previous balanced checkpoint, so a future
interaction-weighted fine-tune should start from this release rather than
replacing the dataset pipeline.

## Files

- Release: `trans1/releases/trans1_dynamic_v2.pt`
- Training run: `trans1/runs/trans1_dynamic_v2`
- Generator: `v2/generate_16m_handflow_dynamic_arranger.py`
- Web model id: `v2.1-handflow`
