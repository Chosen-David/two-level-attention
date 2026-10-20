# TL-ENUM-CONTRACT-20261020

## Plan

目标：四个既有枚举校验在普通和-O模式一致生效，固定配额及legacy合法语义不变。
基线bb173c7；主控拥有源码、测试、证据、advice和TASK。本机host验证，无GPU，
不涉及SGLang。先32反例红证，再最小显式校验，21项普通/-O回归及336控制差分；
发布前fetch，普通main更新，回读SHA/tree。人类guide只读。

## Progress

- 修前-O的32非法子例均接受；修后普通/-O各21测试通过，336合法全属性对照不变。
- E151c三模块因缺Torch导入失败；GPU/模型/性能未验。详见advice
  `2026-10-20_tli_enum_contract_by_gpt.md`，复验脚本在results/tli-enum-contract-20261020。
- 有界host修复完成；发布和远端内容回读由主控收尾。持续优化未完成。
