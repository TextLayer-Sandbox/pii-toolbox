**Table 2: F2 by PII type, relaxed matching**

| Dataset | Tool | PERSON | ORGANIZATION | LOCATION | DATE_TIME | QUANTITY | IDENTIFIER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TAB | gliner | 0.647 | 0.566 | 0.578 | 0.769 | 0.375 | 0.511 |
| TAB | presidio_structured | n/a | n/a | n/a | n/a | n/a | 0.011 |
| TAB | roberta | 0.853 | 0.511 | 0.734 | n/a | n/a | n/a |
| TAB | spacy_lg | 0.853 | 0.479 | 0.651 | 0.952 | 0.201 | n/a |
| TAB | spacy_sm | 0.740 | 0.455 | 0.641 | 0.940 | 0.114 | n/a |
| TAB | gliner+presidio_structured | 0.650 | 0.564 | 0.578 | 0.769 | 0.376 | 0.516 |
| TAB | roberta+presidio_structured | 0.857 | 0.514 | 0.740 | n/a | n/a | 0.011 |
| TAB | spacy_lg+presidio_structured | 0.853 | 0.479 | 0.651 | 0.952 | 0.201 | 0.011 |
| TAB | spacy_sm+presidio_structured | 0.740 | 0.455 | 0.641 | 0.940 | 0.114 | 0.011 |
| MAPA | gliner | 0.543 | 0.255 | 0.000 | 0.828 | 0.000 |  |
| MAPA | presidio_structured | n/a | n/a | n/a | n/a | n/a |  |
| MAPA | roberta | 0.660 | 0.225 | 0.826 | n/a | n/a |  |
| MAPA | spacy_lg | 0.521 | 0.162 | 0.744 | 0.715 | n/a |  |
| MAPA | spacy_sm | 0.516 | 0.155 | 0.769 | 0.642 | 0.000 |  |
| MAPA | gliner+presidio_structured | 0.548 | 0.256 | 0.000 | 0.828 | 0.000 |  |
| MAPA | roberta+presidio_structured | 0.656 | 0.228 | 0.826 | n/a | n/a |  |
| MAPA | spacy_lg+presidio_structured | 0.521 | 0.162 | 0.744 | 0.717 | n/a |  |
| MAPA | spacy_sm+presidio_structured | 0.516 | 0.155 | 0.769 | 0.643 | 0.000 |  |
| Worldwide | gliner | 0.728 | 0.642 | 0.344 | 0.510 | 0.828 |  |
| Worldwide | presidio_structured | n/a | n/a | n/a | n/a | n/a |  |
| Worldwide | roberta | 0.961 | 0.854 | 0.889 | n/a | n/a |  |
| Worldwide | spacy_lg | 0.811 | 0.676 | 0.799 | 0.691 | 0.560 |  |
| Worldwide | spacy_sm | 0.709 | 0.632 | 0.729 | 0.707 | 0.534 |  |
| Worldwide | gliner+presidio_structured | 0.727 | 0.642 | 0.343 | 0.511 | 0.828 |  |
| Worldwide | roberta+presidio_structured | 0.968 | 0.882 | 0.902 | n/a | n/a |  |
| Worldwide | spacy_lg+presidio_structured | 0.812 | 0.676 | 0.799 | 0.690 | 0.560 |  |
| Worldwide | spacy_sm+presidio_structured | 0.710 | 0.632 | 0.730 | 0.706 | 0.534 |  |
