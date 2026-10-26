**Table 6: Cost and speed**

| Tool | Peak memory (GiB) | Load time (s) | Throughput at 1k chars (chars/s) | Throughput at 10k chars (chars/s) | Throughput at 100k chars (chars/s) | Latency per 1,000 chars (s) | fits Lambda (10 GB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gliner | 2.78 | 6.00 | 9992 | 8905 | 8914 | 0.112 | yes |
| presidio_structured | 0.45 | 2.32 | 383454 | 384836 | 394817 | 0.003 | yes |
| roberta | 2.49 | 1.75 | 17845 | 18305 | 18100 | 0.055 | yes |
| spacy_lg | 1.33 | 0.98 | 57266 | 61229 | 60022 | 0.017 | yes |
| spacy_sm | 0.72 | 0.69 | 62586 | 65839 | 64543 | 0.015 | yes |
| gliner+presidio_structured | 2.78 | 6.40 | 9715 | 8764 | 8786 | 0.114 | yes |
| roberta+presidio_structured | 2.63 | 2.36 | 17012 | 17430 | 17249 | 0.058 | yes |
| spacy_lg+presidio_structured | 1.54 | 2.49 | 49386 | 52687 | 51837 | 0.019 | yes |
| spacy_sm+presidio_structured | 0.93 | 2.20 | 53296 | 56086 | 55433 | 0.018 | yes |
