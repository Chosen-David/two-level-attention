# 对比论文原件与链接

实验清单、源码链和阶段计时边界见 [GPT advice](../2026-10-11_benchmark_paper_scope_by_gpt.md)。逐篇作者、版本、DOI、来源、许可、原字节哈希/页数见 [papers.json](papers.json) 与 [独立NOTICE](NOTICE.md)。第三方PDF不继承主库代码许可证。

| 方法 | 固定版 | 本地状态 |
|---|---|---|
| Quest | PMLR235 ICML2024正式版，11页，CC BY4 | [PDF](Quest-PMLR235-2024.pdf)；不是arXiv v2 |
| ClusterKV | [2412.03213v2](https://arxiv.org/abs/2412.03213v2)，CC BY4 | 许可允许，下载被代理403阻塞；仅链接 |
| FASA | [2602.03152v3](https://arxiv.org/abs/2602.03152v3)，CC BY4 | 许可允许，下载被代理403阻塞；仅链接 |
| SparQ | [2312.04985v6](https://arxiv.org/abs/2312.04985v6)，CC BY-SA4 | 许可允许，下载被代理403阻塞；仅链接 |
| MInference | [2407.02490v2](https://arxiv.org/abs/2407.02490v2) | arXiv非独占分发授权不等于本库再分发权，link-only |
| HiP | [ICLR2025正式版](https://proceedings.iclr.cc/paper_files/paper/2025/file/043f0503c4f652c737add3690aa5d12c-Paper-Conference.pdf)，独审46页 | 正式版许可未核；arXiv v3为NC-SA，link-only |
| MoBA | [2502.13189v1](https://arxiv.org/abs/2502.13189v1)，CC BY-NC-ND4 | 未来仓库用途未明确，保守link-only，不声称本项目已确定商业用途 |

优先阅读ClusterKV、Quest、FASA、SparQ的decode与缓存维护，再看MInference的prefill；HiP/MoBA是条件候选。论文页节以advice所标版本为准，尤其Quest正式版与arXiv页码不可互换。未下载成功的PDF没有本地哈希/页数，不以网页可读冒称原件已入库。
