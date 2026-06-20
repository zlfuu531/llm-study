# 代码大模型、Code Agent 与 SWE-bench 面试

这一章面向代码大模型、AI coding、Code Agent、软件工程智能体、算法应用和大模型评测方向。2025-2026 的面试里，代码能力不再只是“HumanEval 能写函数”，而是越来越多追问：

- HumanEval、MBPP、SWE-bench 分别评估什么？
- pass@k 怎么理解？
- 代码补全、代码生成、代码修复有什么区别？
- 代码 Agent 怎么读仓库、改代码、跑测试、修失败？
- 仓库级上下文怎么检索？
- 为什么 benchmark 分数高不等于真实工程可用？
- 代码执行为什么必须沙箱？
- 代码模型数据污染怎么防？

如果被继续追问 pass@k、多候选采样、temperature/top_p 对代码正确率的影响，跳到 [36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md](36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md)。

先背一句：

```text
代码大模型能力 = 代码理解 + 生成 + 补全 + 调试 + 测试 + 仓库级上下文 + 工具闭环。
```

Code Agent 的生产链路可以这样讲：

```text
issue/task
  -> repo understanding
  -> file/symbol retrieval
  -> plan
  -> patch generation
  -> static check / unit tests
  -> failure diagnosis
  -> repair loop
  -> final diff + evidence
```

## 1. 为什么代码方向变高频

代码是 LLM 最容易形成闭环的场景之一，因为它有外部可验证信号：

- 单元测试。
- 编译器。
- linter。
- 类型检查。
- benchmark。
- diff。
- CI 日志。

普通问答很难自动判断“对不对”，但代码可以运行。Code Agent 的核心价值就是把 LLM 的生成能力放进“可执行、可反馈、可修复”的循环里。

面试表达：

> 代码模型不只是写函数，真正的工程场景要能理解仓库、定位文件、生成 patch、运行测试、根据失败日志修复，并控制安全和变更范围。

## 2. 代码模型任务分类

| 任务 | 输入 | 输出 | 关键能力 |
| --- | --- | --- | --- |
| Code completion | 前缀/上下文 | 继续补全 | 局部上下文、语法风格 |
| Text-to-code | 自然语言需求 | 函数/脚本 | 指令理解、算法实现 |
| Code translation | 源语言代码 | 目标语言代码 | 语义保持、库差异 |
| Code summarization | 代码 | 解释/注释 | 代码理解 |
| Bug fixing | 报错/issue/代码 | patch | 定位、修改、验证 |
| Test generation | 函数/需求 | 测试用例 | 边界条件、断言 |
| Repo QA | 仓库 + 问题 | 文件/解释/修改建议 | 仓库级检索 |
| Agentic coding | issue + repo + tools | diff + evidence | 工具使用和闭环 |

面试不要把所有代码能力都说成“代码生成”。补全、修复、测试和仓库级任务的上下文、指标和风险都不同。

## 3. 代码模型训练数据有什么特殊

代码数据和普通文本不同：

- 有明确语法结构。
- 有依赖和项目上下文。
- 有许可证和版权问题。
- 有重复和泄漏风险。
- 有可执行反馈。
- 有注释、文档、commit、issue、PR 等天然配对数据。

常见数据来源：

```text
source code
docstring / comments
README / docs
commit message
issue -> PR diff
unit tests
compiler / linter logs
StackOverflow-like QA
```

清洗重点：

- 去重：同一仓库 fork、镜像、模板代码。
- 许可证过滤：避免不允许训练或商用的数据。
- 秘钥扫描：去除 API key、密码、token。
- PII 清理：作者邮箱、内部地址。
- 质量过滤：空文件、生成文件、压缩代码、vendor 目录。
- 数据污染：过滤 benchmark 测试和答案。

## 4. 代码 tokenizer 为什么特殊

代码里有大量符号、缩进、命名、路径和稀有 token。

需要关注：

- 空格和缩进对 Python 等语言有语义。
- camelCase、snake_case、路径名会被切成多个 subword。
- 括号、冒号、分号、引号、换行都影响语法。
- 长代码文件会产生很长上下文。
- 不同语言 token 分布差异大。

面试表达：

> 代码 tokenizer 要兼顾自然语言和程序语言。切得太碎会增加上下文长度，切得太粗会导致词表大、稀有标识符泛化差。代码补全还要保留缩进和换行。

## 5. 代码补全 vs 代码生成 vs 代码修复

代码补全：

- 输入是当前文件前后缀。
- 强调低延迟、风格一致、语法正确。
- 常见于 IDE。

代码生成：

- 输入是自然语言需求或函数签名。
- 强调算法正确和边界条件。
- 常用 HumanEval、MBPP 类任务评估。

代码修复：

- 输入是 issue、报错、代码仓库、测试失败。
- 输出是 patch。
- 强调定位、最小改动、测试闭环。
- SWE-bench 更接近这一类。

一句话：

> 补全看局部上下文，生成看需求到代码，修复看仓库理解和验证闭环。

## 6. HumanEval、MBPP、SWE-bench 区别

| Benchmark | 看什么 | 局限 |
| --- | --- | --- |
| HumanEval | Python 函数生成，隐藏测试 | 小函数、算法题风格，不代表仓库工程 |
| MBPP | 基础编程题 | 难度较基础，容易被污染 |
| EvalPlus | 给 HumanEval/MBPP 加更强测试 | 仍偏函数级 |
| LiveCodeBench | 时间切分的新题，降低污染 | 主要还是竞赛/函数/短题 |
| BigCodeBench | 更复杂代码生成和库使用 | 仍不是完整真实 repo issue |
| SWE-bench | 真实 GitHub issue 修复 | 更接近工程，但环境构建和评测成本高 |

面试表达：

> HumanEval/MBPP 看函数级生成，SWE-bench 看真实软件 issue 修复。代码 Agent 如果只在 HumanEval 高，不代表能改真实仓库。

## 7. pass@k 怎么理解

代码生成常用 pass@k：模型采样 k 个候选，只要有一个通过测试就算成功。

如果采样 `n` 个候选，其中 `c` 个正确，pass@k 的无偏估计常写作：

```text
pass@k = 1 - C(n - c, k) / C(n, k)
```

直觉：

- `pass@1` 更接近一次生成就对。
- `pass@k` 体现多次采样后是否能碰到正确答案。
- k 越大分数越高，但推理成本也越高。

面试高分点：

> pass@k 不是用户体验的全部。真实 IDE 场景常更关心 pass@1、编辑距离、延迟和是否引入安全问题；Agent 场景还要看测试闭环和 patch 质量。

## 8. 为什么代码评测容易被污染

代码 benchmark 很容易出现在 GitHub、博客、教程和训练数据里。如果模型训练时见过题目或答案，分数会虚高。

污染来源：

- benchmark 原题进入训练集。
- hidden tests 泄露。
- 题解仓库进入训练集。
- 评测题被论坛/博客广泛转载。
- 数据去重只做 exact match，没做近似去重。

治理：

- 时间切分：只用训练截止后发布的新题。
- 去重：文件级、函数级、n-gram、AST、embedding 近似去重。
- 私有测试：隐藏更多边界 case。
- 动态评测：持续加入新 issue、新题。
- 报告时区分公开 benchmark 和私有 eval。

## 9. 仓库级代码理解怎么做

真实代码修复不是一个函数，而是一整个仓库。

需要构建 repo context：

```text
repo tree
  -> README / docs
  -> dependency files
  -> symbol index
  -> imports / call graph
  -> related files
  -> tests
  -> issue text / stack trace
```

常见检索方式：

- 关键词搜索：函数名、类名、错误信息。
- embedding 检索：语义相关文件/片段。
- AST/symbol index：定义、引用、调用关系。
- git history：相关 commit、PR、issue。
- test discovery：找到对应测试文件。

面试表达：

> 仓库级代码 Agent 的关键不是把整个 repo 塞进上下文，而是通过文件树、符号索引、搜索、依赖关系和测试定位，把最相关上下文放进去。

## 10. Code RAG 和普通文档 RAG 的区别

| 维度 | 文档 RAG | Code RAG |
| --- | --- | --- |
| 基本单元 | 段落、chunk、页 | 文件、函数、类、symbol、测试 |
| 检索信号 | 语义、关键词、metadata | 语义、符号、import、调用图、错误栈 |
| 正确性 | 证据支持 | 语法、类型、测试、行为一致 |
| 输出 | 自然语言答案 | patch、diff、代码解释 |
| 风险 | 幻觉、引用错 | 破坏接口、引入 bug、安全漏洞 |

Code RAG 要特别重视：

- 代码边界：函数、类、模块，而不是随便按长度切。
- import 依赖。
- 调用关系。
- 测试文件。
- 版本和分支。

## 11. Code Agent 的最小闭环

最小闭环：

```text
read issue
  -> inspect repo
  -> locate files
  -> propose plan
  -> edit patch
  -> run tests
  -> inspect failure
  -> revise patch
  -> final answer with evidence
```

核心不是“生成代码”，而是：

- 是否能找到该改哪里。
- 是否能做最小修改。
- 是否能运行验证。
- 是否能读懂失败日志。
- 是否能避免改坏无关行为。

面试可以说：

> Code Agent 的优势是有编译和测试这种可验证反馈，所以它应该把生成、执行、诊断、修复做成循环。

## 12. Patch 生成和 diff 约束

真实工程里最好输出 patch/diff，而不是整文件重写。

好处：

- 改动范围可审查。
- 容易回滚。
- 不容易破坏无关代码。
- 方便 CI 和 code review。

Patch 约束：

- 尽量最小修改。
- 不改无关格式。
- 保留公共 API。
- 补充或更新测试。
- 避免硬编码测试答案。
- 避免删除失败测试来“通过”。

面试高分点：代码 Agent 不能为了过测试而作弊，比如改测试、跳过测试、mock 掉真实逻辑。

## 13. 测试闭环怎么设计

测试分层：

```text
unit tests
integration tests
type check
lint
format
build
security scan
```

执行策略：

- 先跑与修改文件相关的局部测试。
- 失败后读取日志定位。
- 修完再跑更大范围测试。
- 有时间预算时跑全量。
- 记录命令、退出码和关键日志。

面试表达：

> 我会让 Agent 优先跑最小相关测试来快速迭代，最后再跑更大范围回归。每次 patch 都要和测试证据绑定，而不是只看模型自评。

## 14. 代码执行沙箱和安全

代码模型输出不能直接在生产环境执行。

风险：

- 恶意命令。
- 删除文件。
- 读取密钥。
- 网络外连。
- 无限循环。
- 高资源消耗。
- 供应链攻击。
- prompt injection 让 Agent 执行危险命令。

沙箱措施：

```text
container / VM
no secrets
limited filesystem
network off or allowlist
CPU/memory/time limit
read-only source snapshot when possible
audit log
manual approval for high-risk commands
```

面试要强调：测试环境也不能随便给生产密钥。

## 15. SWE-bench 为什么更接近真实工程

SWE-bench 来自真实 GitHub issue，要求模型根据 issue 修改仓库，使测试通过。它比 HumanEval 更接近真实软件工程，因为要：

- 理解 issue。
- 找相关文件。
- 修改已有代码。
- 保持接口兼容。
- 运行测试。
- 处理真实依赖和环境。

局限：

- 环境构建复杂。
- 测试不一定覆盖所有正确性。
- 模型可能过拟合公开 issue。
- 评分通过不等于 patch 质量完美。
- 很多真实业务需求没有现成测试。

面试表达：

> SWE-bench 的价值是把代码能力从函数生成推进到真实仓库修复，但它仍然只是评估维度之一，不能替代 code review、安全审查和私有项目 eval。

## 16. 代码 Agent Eval 指标

除了 pass@k，还要看：

- issue resolved rate。
- test pass rate。
- patch apply success rate。
- regression rate。
- average iterations。
- time to fix。
- changed lines / files。
- human review acceptance。
- security violation rate。
- build success rate。
- cost per issue。

轨迹指标：

```text
找到正确文件了吗？
是否理解错误日志？
是否运行了正确测试？
是否改了无关文件？
是否补测试？
是否有危险命令？
```

## 17. 代码模型和普通 LLM 的差异

代码模型通常更重视：

- 代码语法和缩进。
- 多语言数据。
- 代码-注释对齐。
- 仓库上下文。
- 长上下文和 fill-in-the-middle。
- 单测反馈。
- API 和库使用。

Fill-in-the-middle 常见于 IDE 补全：

```text
prefix + <hole> + suffix -> middle code
```

相比只根据前缀续写，FIM 能利用光标后面的代码约束生成内容。

## 18. 常见面试误区

误区 1：只讲 HumanEval 分数。  
更好：区分函数级、竞赛级、仓库级、Agentic coding。

误区 2：说“跑测试就行”。  
更好：讲测试选择、失败日志诊断、回归、沙箱和 review。

误区 3：把整个仓库塞进长上下文。  
更好：讲 repo tree、symbol index、search、call graph、test discovery 和 context budget。

误区 4：代码 Agent 改得越多越好。  
更好：最小 patch、可审查、可回滚。

误区 5：忽略安全。  
更好：命令白名单、容器、无密钥、资源限制、网络限制和审计。

## 19. 项目 8 分钟讲法

```text
背景：
我们做的是代码智能体/代码修复系统，目标是根据 issue 或测试失败自动定位代码、生成 patch，并用测试验证。

架构：
Issue/Task -> Repo Index -> File/Symbol Retrieval -> Planner
-> Patch Generator -> Test Runner -> Failure Analyzer -> Repair Loop
-> Final Diff + Evidence

关键设计：
1. 对 repo 建文件树、符号索引、依赖关系和测试映射。
2. 用关键词、embedding、AST/symbol 搜索定位相关文件。
3. 让模型生成最小 diff，而不是整文件重写。
4. 在沙箱里跑单测、lint、type check 和 build。
5. 根据失败日志做二次修复，并限制最大迭代次数。
6. 记录 trace：检索到的文件、patch、测试命令、失败日志、最终证据。

指标：
issue resolved rate、test pass rate、patch apply rate、平均迭代次数、改动行数、人工 review 通过率、成本和安全违规率。

难点：
仓库上下文太大、依赖环境难构建、测试不完整、模型可能改无关文件、失败日志理解不稳定、执行环境要隔离。
```

## 20. 面试前背诵版

代码大模型能力不只是 HumanEval 写函数，而是代码理解、补全、生成、修复、测试和仓库级上下文。HumanEval/MBPP 偏函数级代码生成，SWE-bench 更接近真实 GitHub issue 修复。pass@k 表示采样 k 个候选里至少一个通过测试的概率，但真实工程还要看 pass@1、patch 质量、测试闭环、安全和 review。Code Agent 的核心链路是读 issue、检索 repo、定位文件、生成最小 patch、运行测试、根据失败日志修复，并在沙箱里控制权限和资源。仓库级上下文不能全塞给模型，要用文件树、符号索引、搜索、调用关系和测试映射做 context selection。

## 本轮参考

- HumanEval 论文：[https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)
- HumanEval GitHub：[https://github.com/openai/human-eval](https://github.com/openai/human-eval)
- MBPP 论文：[https://arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732)
- APPS 论文：[https://arxiv.org/abs/2105.09938](https://arxiv.org/abs/2105.09938)
- CodeContests 论文：[https://arxiv.org/abs/2203.07814](https://arxiv.org/abs/2203.07814)
- SWE-bench 论文：[https://arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)
- SWE-bench 官网：[https://www.swebench.com/](https://www.swebench.com/)
- EvalPlus 论文：[https://arxiv.org/abs/2305.01210](https://arxiv.org/abs/2305.01210)
- LiveCodeBench 论文：[https://arxiv.org/abs/2403.07974](https://arxiv.org/abs/2403.07974)
- BigCodeBench 论文：[https://arxiv.org/abs/2406.15877](https://arxiv.org/abs/2406.15877)
- RepoBench 论文：[https://arxiv.org/abs/2306.03091](https://arxiv.org/abs/2306.03091)
- Code Llama 论文：[https://arxiv.org/abs/2308.12950](https://arxiv.org/abs/2308.12950)
- StarCoder2 论文：[https://arxiv.org/abs/2402.19173](https://arxiv.org/abs/2402.19173)
- DeepSeek-Coder 论文：[https://arxiv.org/abs/2401.14196](https://arxiv.org/abs/2401.14196)
- Qwen2.5-Coder 技术报告：[https://arxiv.org/abs/2409.12186](https://arxiv.org/abs/2409.12186)
