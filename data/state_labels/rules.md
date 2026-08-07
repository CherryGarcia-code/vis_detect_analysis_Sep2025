# Behavioral-state rule

window W = 15
LOSO Cohen's kappa = 0.709
states = ['Abort', 'Disengaged', 'Impulsive', 'StimSens']

```
|--- f_nolick <= 0.41
|   |--- f_abort <= 0.38
|   |   |--- f_inapplick <= 0.45
|   |   |   |--- class: StimSens
|   |   |--- f_inapplick >  0.45
|   |   |   |--- class: Impulsive
|   |--- f_abort >  0.38
|   |   |--- f_abort <= 0.52
|   |   |   |--- class: Abort
|   |   |--- f_abort >  0.52
|   |   |   |--- class: Abort
|--- f_nolick >  0.41
|   |--- f_applick <= 0.24
|   |   |--- f_abort <= 0.32
|   |   |   |--- class: Disengaged
|   |   |--- f_abort >  0.32
|   |   |   |--- class: Disengaged
|   |--- f_applick >  0.24
|   |   |--- f_abort <= 0.03
|   |   |   |--- class: Disengaged
|   |   |--- f_abort >  0.03
|   |   |   |--- class: StimSens

```
