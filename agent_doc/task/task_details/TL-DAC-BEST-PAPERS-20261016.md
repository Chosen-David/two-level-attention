# TL-DAC-BEST-PAPERS-20261016

来源：用户 2026-10-16 明确要求在 two-level-attention 的 advice 下创建
DAC_best_paper，收集近十年 DAC Best Paper PDF，并直接推送 main。

## 已完成

- 按 2017–2026 十届核实 Research Best Paper 获奖名单，共 13 篇。
- 创建 [DAC_best_paper](../../advice/DAC_best_paper/README.md)，保存 7 篇原始 PDF。
- 逐份验证 PDF 头、页数、全文可解析及首页标题；记录来源、版本、SHA-256。
- 2023 年两篇均按 ACM SIGDA 同期公告确认；DREAMPlace 使用 DAC 2019 会议版。
- 下载文件为论文原始字节，未改图文、未生成替代全文。

## 未完成 / 阻塞

| 年份 | 缺失论文 |
| --- | --- |
| 2017 | Toward Optimal Legalization for Mixed-Cell-Height Circuit Designs |
| 2017 | Low-Power On-Chip Network Providing Guaranteed Services for Snoopy Coherent and Artificial Neural Network Systems |
| 2018 | Compensated-DNN |
| 2018 | Exact Algorithms for Delay-Bounded Steiner Arborescences |
| 2022 | Towards a Formally Verified Hardware Root-of-Trust for Data-Oblivious Computing |
| 2025 | INSTA |

前五篇出版商全文请求返回 HTTP 403/418，作者主页和 OpenAlex 等公开来源
未找到可用全文。INSTA 的 NVIDIA PDF 地址返回 Git LFS 指针（132 字节，
对象预期 1451676 字节），公开 NVlabs/INSTA 源码树未附 PDF。未把错误页、
指针、会议日程、报告幻灯片或期刊扩展版冒充获奖论文。

## 下一步与验收

需要上述六篇的可用公开全文、用户提供的 PDF 或可用的机构全文访问。
补齐时检查论文身份、原始版本、全部页可解析、SHA-256，并同步 papers.json、
SHA256SUMS 和 README 计数。所有 13 篇全文实际入库后才关闭本任务。
本次发布为已取得材料与完整索引的部分交付，不表示已全部下载。
