**Table 4: Co-reference off versus on (English text, strict matching)**

| Dataset | Detector | F2 off | F2 on | F2 change | ER_di off | ER_di on | ER_di change | ER_qi off | ER_qi on | ER_qi change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TAB | gliner | 0.644 | 0.650 | +0.005 | 0.803 | 0.829 | +0.026 | 0.675 | 0.682 | +0.008 |
| TAB | presidio_structured | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| TAB | roberta | 0.373 | 0.373 | -0.000 | 0.485 | 0.485 | +0.000 | 0.251 | 0.251 | +0.000 |
| TAB | spacy_lg | 0.564 | 0.567 | +0.003 | 0.489 | 0.490 | +0.001 | 0.801 | 0.804 | +0.003 |
| TAB | spacy_sm | 0.536 | 0.540 | +0.004 | 0.454 | 0.458 | +0.004 | 0.780 | 0.783 | +0.003 |
| TAB | gliner+presidio_structured | 0.645 | 0.651 | +0.005 | 0.803 | 0.829 | +0.026 | 0.675 | 0.682 | +0.008 |
| TAB | roberta+presidio_structured | 0.344 | 0.344 | -0.000 | 0.485 | 0.485 | +0.000 | 0.251 | 0.251 | +0.000 |
| TAB | spacy_lg+presidio_structured | 0.539 | 0.542 | +0.003 | 0.489 | 0.490 | +0.001 | 0.801 | 0.804 | +0.003 |
| TAB | spacy_sm+presidio_structured | 0.512 | 0.516 | +0.004 | 0.454 | 0.458 | +0.004 | 0.780 | 0.783 | +0.003 |
| MAPA | gliner | 0.492 | 0.506 | +0.014 |  |  |  |  |  |  |
| MAPA | presidio_structured | 0.000 | 0.000 | +0.000 |  |  |  |  |  |  |
| MAPA | roberta | 0.314 | 0.313 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_lg | 0.378 | 0.377 | -0.000 |  |  |  |  |  |  |
| MAPA | spacy_sm | 0.332 | 0.337 | +0.005 |  |  |  |  |  |  |
| MAPA | gliner+presidio_structured | 0.494 | 0.507 | +0.014 |  |  |  |  |  |  |
| MAPA | roberta+presidio_structured | 0.314 | 0.313 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_lg+presidio_structured | 0.378 | 0.377 | -0.000 |  |  |  |  |  |  |
| MAPA | spacy_sm+presidio_structured | 0.332 | 0.337 | +0.005 |  |  |  |  |  |  |
| Worldwide | gliner | 0.507 | 0.542 | +0.035 |  |  |  |  |  |  |
| Worldwide | presidio_structured | 0.000 | 0.000 | +0.000 |  |  |  |  |  |  |
| Worldwide | roberta | 0.873 | 0.872 | -0.001 |  |  |  |  |  |  |
| Worldwide | spacy_lg | 0.645 | 0.648 | +0.003 |  |  |  |  |  |  |
| Worldwide | spacy_sm | 0.585 | 0.596 | +0.010 |  |  |  |  |  |  |
| Worldwide | gliner+presidio_structured | 0.505 | 0.541 | +0.035 |  |  |  |  |  |  |
| Worldwide | roberta+presidio_structured | 0.887 | 0.886 | -0.001 |  |  |  |  |  |  |
| Worldwide | spacy_lg+presidio_structured | 0.645 | 0.649 | +0.003 |  |  |  |  |  |  |
| Worldwide | spacy_sm+presidio_structured | 0.586 | 0.596 | +0.010 |  |  |  |  |  |  |
