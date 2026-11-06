**Table 2: F2 by PII type, strict matching**

| Dataset | Tool | PERSON | ORGANIZATION | LOCATION | DATE_TIME | QUANTITY | IDENTIFIER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TAB | gliner | 0.636 | 0.519 | 0.563 | 0.767 | 0.366 | 0.501 |
| TAB | presidio_structured | n/a | n/a | n/a | n/a | n/a | 0.000 |
| TAB | roberta | 0.129 | 0.431 | 0.697 | n/a | n/a | n/a |
| TAB | spacy_lg | 0.337 | 0.141 | 0.561 | 0.912 | 0.120 | n/a |
| TAB | spacy_sm | 0.282 | 0.116 | 0.559 | 0.889 | 0.067 | n/a |
| TAB | gliner+presidio_structured | 0.639 | 0.519 | 0.563 | 0.767 | 0.367 | 0.499 |
| TAB | roberta+presidio_structured | 0.128 | 0.434 | 0.700 | n/a | n/a | 0.000 |
| TAB | spacy_lg+presidio_structured | 0.337 | 0.141 | 0.561 | 0.912 | 0.120 | 0.000 |
| TAB | spacy_sm+presidio_structured | 0.282 | 0.116 | 0.559 | 0.889 | 0.067 | 0.000 |
| MAPA | gliner | 0.543 | 0.245 | 0.000 | 0.828 | 0.000 |  |
| MAPA | presidio_structured | n/a | n/a | n/a | n/a | n/a |  |
| MAPA | roberta | 0.460 | 0.176 | 0.780 | n/a | n/a |  |
| MAPA | spacy_lg | 0.291 | 0.095 | 0.702 | 0.707 | n/a |  |
| MAPA | spacy_sm | 0.312 | 0.074 | 0.726 | 0.566 | 0.000 |  |
| MAPA | gliner+presidio_structured | 0.548 | 0.246 | 0.000 | 0.828 | 0.000 |  |
| MAPA | roberta+presidio_structured | 0.453 | 0.178 | 0.780 | n/a | n/a |  |
| MAPA | spacy_lg+presidio_structured | 0.291 | 0.095 | 0.702 | 0.708 | n/a |  |
| MAPA | spacy_sm+presidio_structured | 0.312 | 0.074 | 0.726 | 0.567 | 0.000 |  |
| Worldwide | gliner | 0.631 | 0.553 | 0.322 | 0.498 | 0.809 |  |
| Worldwide | presidio_structured | n/a | n/a | n/a | n/a | n/a |  |
| Worldwide | roberta | 0.935 | 0.812 | 0.865 | n/a | n/a |  |
| Worldwide | spacy_lg | 0.762 | 0.445 | 0.733 | 0.636 | 0.279 |  |
| Worldwide | spacy_sm | 0.653 | 0.404 | 0.672 | 0.649 | 0.230 |  |
| Worldwide | gliner+presidio_structured | 0.629 | 0.551 | 0.321 | 0.499 | 0.806 |  |
| Worldwide | roberta+presidio_structured | 0.940 | 0.837 | 0.876 | n/a | n/a |  |
| Worldwide | spacy_lg+presidio_structured | 0.762 | 0.445 | 0.733 | 0.635 | 0.279 |  |
| Worldwide | spacy_sm+presidio_structured | 0.654 | 0.405 | 0.672 | 0.649 | 0.230 |  |
