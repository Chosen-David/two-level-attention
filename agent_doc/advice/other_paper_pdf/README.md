# 对比论文原件与链接

实验清单、源码链和阶段计时边界见 [GPT advice](../2026-10-11_benchmark_paper_scope_by_gpt.md)。逐篇作者、版本、DOI、来源、许可、原字节哈希/页数见 [papers.json](papers.json) 与 [独立NOTICE](NOTICE.md)。第三方PDF不继承主库代码许可证。

| 方法 | 固定版 | 本地状态 |
|---|---|---|
| Quest | PMLR235 ICML2024正式版，11页，CC BY4 | [PDF](Quest-PMLR235-2024.pdf)；不是arXiv v2 |
| ClusterKV | [2412.03213v2](https://arxiv.org/abs/2412.03213v2)，CC BY4 | 许可允许，下载被代理403阻塞；仅链接 |
| FASA | [2602.03152v3](https://arxiv.org/abs/2602.03152v3)，CC BY4 | 许可允许，下载被代理403阻塞；仅链接 |
| SparQ | PMLR235 ICML2024正式版，26页，CC BY4 | [PDF](SparQ-PMLR235-2024.pdf)；[arXiv v6](https://arxiv.org/abs/2312.04985v6)为不同的27页BY-SA4版，仅链接；advice中v6页节不对应此PDF |
| MInference | [2407.02490v2](https://arxiv.org/abs/2407.02490v2) | arXiv非独占分发授权不等于本库再分发权，link-only |
| HiP | [ICLR2025正式版](https://proceedings.iclr.cc/paper_files/paper/2025/file/043f0503c4f652c737add3690aa5d12c-Paper-Conference.pdf)，独审46页 | 正式版许可未核；arXiv v3为NC-SA，link-only |
| MoBA | [2502.13189v1](https://arxiv.org/abs/2502.13189v1)，CC BY-NC-ND4 | 已授权非商业参考、原样独立许可归档；下载403，PDF待补 |
| StreamingLLM | [2309.17453v4](https://arxiv.org/abs/2309.17453v4)，CC BY4 | [官方PDF](https://arxiv.org/pdf/2309.17453v4)；下载403，PDF待补 |
| HISA | [2603.28458v4](https://arxiv.org/abs/2603.28458v4)，CC BY4 | [官方PDF](https://arxiv.org/pdf/2603.28458v4)；下载403，PDF待补 |
| H2O | [2306.14048v3](https://arxiv.org/abs/2306.14048v3) | [官方PDF](https://arxiv.org/pdf/2306.14048v3)；再分发权未验证，仅链接 |

优先阅读ClusterKV、Quest、FASA、SparQ的decode与缓存维护，再看MInference的prefill；HiP/MoBA是条件候选。论文页节以advice所标版本为准，尤其Quest正式版与arXiv页码不可互换。未下载成功的PDF没有本地哈希/页数，不以网页可读冒称原件已入库。

## 2026-10-15 原件补充请求

只更新资料目录，未改代码或运行GPU。资料包 `reference_pdf_bundle_20261015.zip`
的提供方声明 SHA256 为 `6944f2b985c52c9e4012302e6bf23aca53c78573bb5a30c9dc7e03dc0f80be4a`；
本执行器 Library 下载失败，**未本地核验此hash或包内文件**。随后逐一尝试三个
固定版本的官方 arXiv PDF，均被网络代理 CONNECT 403 拒绝；未绕过拒绝。
因此本轮交付链接和许可/作者元数据，三篇原始 PDF 仍未完成归档，不能称已推送PDF。
H2O始终仅链接。后续取得原件后须核PDF文件头、作者/版本、逐文件SHA256/页数，
原样保存，并更新papers.json与NOTICE；MoBA限非商业参考且独立许可，不重许可。
