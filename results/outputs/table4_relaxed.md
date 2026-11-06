**Table 4: Co-reference off versus on (English text, relaxed matching)**

| Dataset | Detector | F2 off | F2 on | F2 change | ER_di off | ER_di on | ER_di change | ER_qi off | ER_qi on | ER_qi change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TAB | gliner | 0.659 | 0.664 | +0.006 | 0.803 | 0.829 | +0.026 | 0.675 | 0.682 | +0.008 |
| TAB | presidio_structured | 0.011 | 0.011 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 |
| TAB | roberta | 0.678 | 0.678 | -0.000 | 0.485 | 0.485 | +0.000 | 0.251 | 0.251 | +0.000 |
| TAB | spacy_lg | 0.768 | 0.771 | +0.003 | 0.489 | 0.490 | +0.001 | 0.801 | 0.804 | +0.003 |
| TAB | spacy_sm | 0.732 | 0.738 | +0.005 | 0.454 | 0.458 | +0.004 | 0.780 | 0.783 | +0.003 |
| TAB | gliner+presidio_structured | 0.659 | 0.665 | +0.006 | 0.803 | 0.829 | +0.026 | 0.675 | 0.682 | +0.008 |
| TAB | roberta+presidio_structured | 0.627 | 0.627 | -0.000 | 0.485 | 0.485 | +0.000 | 0.251 | 0.251 | +0.000 |
| TAB | spacy_lg+presidio_structured | 0.734 | 0.737 | +0.003 | 0.489 | 0.490 | +0.001 | 0.801 | 0.804 | +0.003 |
| TAB | spacy_sm+presidio_structured | 0.700 | 0.705 | +0.005 | 0.454 | 0.458 | +0.004 | 0.780 | 0.783 | +0.003 |
| MAPA | gliner | 0.495 | 0.509 | +0.014 |  |  |  |  |  |  |
| MAPA | presidio_structured | 0.000 | 0.000 | +0.000 |  |  |  |  |  |  |
| MAPA | roberta | 0.405 | 0.404 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_lg | 0.453 | 0.453 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_sm | 0.430 | 0.435 | +0.005 |  |  |  |  |  |  |
| MAPA | gliner+presidio_structured | 0.497 | 0.511 | +0.014 |  |  |  |  |  |  |
| MAPA | roberta+presidio_structured | 0.407 | 0.406 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_lg+presidio_structured | 0.453 | 0.453 | -0.001 |  |  |  |  |  |  |
| MAPA | spacy_sm+presidio_structured | 0.430 | 0.435 | +0.005 |  |  |  |  |  |  |
| Worldwide | gliner | 0.568 | 0.606 | +0.038 |  |  |  |  |  |  |
| Worldwide | presidio_structured | 0.000 | 0.000 | +0.000 |  |  |  |  |  |  |
| Worldwide | roberta | 0.903 | 0.902 | -0.001 |  |  |  |  |  |  |
| Worldwide | spacy_lg | 0.751 | 0.755 | +0.003 |  |  |  |  |  |  |
| Worldwide | spacy_sm | 0.691 | 0.703 | +0.011 |  |  |  |  |  |  |
| Worldwide | gliner+presidio_structured | 0.567 | 0.605 | +0.038 |  |  |  |  |  |  |
| Worldwide | roberta+presidio_structured | 0.918 | 0.918 | -0.001 |  |  |  |  |  |  |
| Worldwide | spacy_lg+presidio_structured | 0.752 | 0.755 | +0.003 |  |  |  |  |  |  |
| Worldwide | spacy_sm+presidio_structured | 0.692 | 0.703 | +0.012 |  |  |  |  |  |  |
