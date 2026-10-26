**Table 2: F2 by PII type, relaxed matching**

| Dataset | Tool | PERSON | ORGANIZATION | LOCATION | DATE_TIME | QUANTITY | IDENTIFIER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TAB | gliner | 0.628 | 0.551 | 0.639 | n/a | n/a | n/a |
| TAB | presidio_structured | n/a | n/a | n/a | n/a | n/a | 0.010 |
| TAB | roberta | 0.888 | 0.530 | 0.739 | n/a | n/a | n/a |
| TAB | spacy_lg | 0.829 | 0.498 | 0.641 | 0.928 | 0.173 | n/a |
| TAB | spacy_sm | 0.728 | 0.478 | 0.626 | 0.921 | 0.095 | n/a |
| TAB | gliner+presidio_structured | 0.630 | 0.550 | 0.643 | n/a | n/a | 0.010 |
| TAB | roberta+presidio_structured | 0.892 | 0.531 | 0.744 | n/a | n/a | 0.010 |
| TAB | spacy_lg+presidio_structured | 0.829 | 0.498 | 0.641 | 0.928 | 0.173 | 0.010 |
| TAB | spacy_sm+presidio_structured | 0.728 | 0.478 | 0.626 | 0.921 | 0.095 | 0.010 |
| MAPA | gliner | 0.601 | 0.285 | 0.056 | n/a | n/a |  |
| MAPA | presidio_structured | n/a | n/a | n/a | n/a | n/a |  |
| MAPA | roberta | 0.736 | 0.253 | 0.826 | n/a | n/a |  |
| MAPA | spacy_lg | 0.521 | 0.216 | 0.744 | 0.724 | n/a |  |
| MAPA | spacy_sm | 0.531 | 0.181 | 0.769 | 0.657 | 0.000 |  |
| MAPA | gliner+presidio_structured | 0.606 | 0.285 | 0.056 | n/a | n/a |  |
| MAPA | roberta+presidio_structured | 0.734 | 0.256 | 0.826 | n/a | n/a |  |
| MAPA | spacy_lg+presidio_structured | 0.521 | 0.216 | 0.744 | 0.725 | n/a |  |
| MAPA | spacy_sm+presidio_structured | 0.531 | 0.181 | 0.769 | 0.658 | 0.000 |  |
