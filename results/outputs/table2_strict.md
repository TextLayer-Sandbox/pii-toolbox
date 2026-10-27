**Table 2: F2 by PII type, strict matching**

| Dataset | Tool | PERSON | ORGANIZATION | LOCATION | DATE_TIME | QUANTITY | IDENTIFIER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TAB | gliner | 0.615 | 0.484 | 0.611 | n/a | n/a | n/a |
| TAB | presidio_structured | n/a | n/a | n/a | n/a | n/a | 0.000 |
| TAB | roberta | 0.124 | 0.418 | 0.662 | n/a | n/a | n/a |
| TAB | spacy_lg | 0.321 | 0.135 | 0.531 | 0.872 | 0.096 | n/a |
| TAB | spacy_sm | 0.269 | 0.111 | 0.528 | 0.849 | 0.053 | n/a |
| TAB | gliner+presidio_structured | 0.616 | 0.484 | 0.612 | n/a | n/a | 0.000 |
| TAB | roberta+presidio_structured | 0.124 | 0.421 | 0.664 | n/a | n/a | 0.000 |
| TAB | spacy_lg+presidio_structured | 0.321 | 0.135 | 0.531 | 0.872 | 0.096 | 0.000 |
| TAB | spacy_sm+presidio_structured | 0.269 | 0.111 | 0.528 | 0.850 | 0.053 | 0.000 |
| MAPA | gliner | 0.559 | 0.243 | 0.056 | n/a | n/a |  |
| MAPA | presidio_structured | n/a | n/a | n/a | n/a | n/a |  |
| MAPA | roberta | 0.460 | 0.176 | 0.780 | n/a | n/a |  |
| MAPA | spacy_lg | 0.291 | 0.095 | 0.702 | 0.707 | n/a |  |
| MAPA | spacy_sm | 0.312 | 0.074 | 0.726 | 0.566 | 0.000 |  |
| MAPA | gliner+presidio_structured | 0.563 | 0.243 | 0.056 | n/a | n/a |  |
| MAPA | roberta+presidio_structured | 0.453 | 0.178 | 0.780 | n/a | n/a |  |
| MAPA | spacy_lg+presidio_structured | 0.291 | 0.095 | 0.702 | 0.708 | n/a |  |
| MAPA | spacy_sm+presidio_structured | 0.312 | 0.074 | 0.726 | 0.567 | 0.000 |  |
| Worldwide | gliner | 0.628 | 0.545 | 0.325 | n/a | n/a |  |
| Worldwide | presidio_structured | n/a | n/a | n/a | n/a | n/a |  |
| Worldwide | roberta | 0.935 | 0.812 | 0.865 | n/a | n/a |  |
| Worldwide | spacy_lg | 0.762 | 0.445 | 0.733 | 0.636 | 0.279 |  |
| Worldwide | spacy_sm | 0.653 | 0.404 | 0.672 | 0.649 | 0.230 |  |
| Worldwide | gliner+presidio_structured | 0.626 | 0.543 | 0.323 | n/a | n/a |  |
| Worldwide | roberta+presidio_structured | 0.940 | 0.837 | 0.876 | n/a | n/a |  |
| Worldwide | spacy_lg+presidio_structured | 0.762 | 0.445 | 0.733 | 0.635 | 0.279 |  |
| Worldwide | spacy_sm+presidio_structured | 0.654 | 0.405 | 0.672 | 0.649 | 0.230 |  |
