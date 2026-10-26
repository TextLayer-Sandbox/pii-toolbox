**Table 4: Co-reference off versus on (English text, relaxed matching)**

| Dataset | Detector | F2 off | F2 on | F2 change | ER_di off | ER_di on | ER_di change | ER_qi off | ER_qi on | ER_qi change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TAB | gliner | 0.329 | 0.336 | +0.007 | 0.466 | 0.490 | +0.024 | 0.236 | 0.244 | +0.008 |
| TAB | presidio_structured | 0.001 | 0.001 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| TAB | roberta | 0.380 | 0.380 | -0.000 | 0.485 | 0.485 | +0.000 | 0.251 | 0.251 | +0.000 |
| TAB | spacy_lg | 0.719 | 0.722 | +0.003 | 0.489 | 0.490 | +0.001 | 0.801 | 0.804 | +0.003 |
| TAB | spacy_sm | 0.690 | 0.695 | +0.005 | 0.454 | 0.458 | +0.004 | 0.780 | 0.783 | +0.003 |
| TAB | gliner+presidio_structured | 0.329 | 0.336 | +0.007 | 0.466 | 0.490 | +0.024 | 0.236 | 0.244 | +0.008 |
| TAB | roberta+presidio_structured | 0.381 | 0.380 | -0.000 | 0.485 | 0.485 | +0.000 | 0.251 | 0.251 | +0.000 |
| TAB | spacy_lg+presidio_structured | 0.719 | 0.722 | +0.003 | 0.489 | 0.490 | +0.001 | 0.801 | 0.804 | +0.003 |
| TAB | spacy_sm+presidio_structured | 0.690 | 0.695 | +0.005 | 0.454 | 0.458 | +0.004 | 0.780 | 0.783 | +0.003 |
| MAPA | gliner | 0.275 | 0.292 | +0.017 |  |  |  |  |  |  |
| MAPA | presidio_structured | 0.000 | 0.000 | +0.000 |  |  |  |  |  |  |
| MAPA | roberta | 0.336 | 0.335 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_lg | 0.478 | 0.478 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_sm | 0.449 | 0.454 | +0.005 |  |  |  |  |  |  |
| MAPA | gliner+presidio_structured | 0.275 | 0.292 | +0.017 |  |  |  |  |  |  |
| MAPA | roberta+presidio_structured | 0.335 | 0.335 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_lg+presidio_structured | 0.478 | 0.477 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_sm+presidio_structured | 0.449 | 0.453 | +0.005 |  |  |  |  |  |  |
