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

## 2026-10-16 用户要求继续查找：第二轮结果

完成六篇的作者/机构目录、公开 GitHub 书目/代码库、OpenAlex、Semantic Scholar
及 INSTA Internet Archive 快照补查，并实际请求六个 IEEE 全文入口，均为 HTTP 418。
本轮未新增有效 PDF，仍为 7/13；维持未完成状态。
NTU 公开书目只链接 IEEE，Bonn 条目只链接 ACM，RPTU 条目未附全文；
INSTA 合作者主页未附 PDF，第一作者链接 IEEE，归档查询无快照，官网 PDF 重试超时。
新检查结果写入 DAC_best_paper/README.md 和 papers.json；现有七个 PDF 字节未修改。

## 2026-10-16 第三轮：arXiv 与作者关键词

用户要求尝试标题、作者加关键词和预印本渠道后继续推送。六篇分别执行 arXiv API
及网页检索；未获得可确认的获奖原文。下载并核验同作者后续期刊论文
A Scalable Formal Verification Methodology for Data-Oblivious Hardware
（arXiv:2308.07757v2，14 页），存入 DAC_best_paper/related/，并记录
作者仓库对其与 DAC 2022 原文关系的说明。原文仍为 7/13，缺 6 篇；
补充阅读 1 篇单独计数，任务保持未完成。来源、版本、校验和及索引已同步。

## 2026-10-16 第四轮：继续寻找全文

补查 NTU 存储库、CEDA 目录 PDF、Crossref、Consensus、GitHub 精确 DOI 和作者主页。
CEDA 文件是目录，不是论文；新入口未获得六篇中的任何原文。ACM 浏览器遇反复安全验证，
已报告并停止尝试；INSTA 官网页面再次返回 132 字节 Git LFS 指针。
整理 AUTHOR_REQUESTS.md，逐篇列出公开作者联系方式、来源及完整索取邮件；均未发送，
需要用户明确授权后才能对外联系。当前仍为 7/13 原文及 1 篇后续研究；任务不关闭。
