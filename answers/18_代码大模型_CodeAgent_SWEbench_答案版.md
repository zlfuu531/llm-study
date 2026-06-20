# 答案版 18：代码大模型、Code Agent 与 SWE-bench

对应题号：381-400。建议先读 [23_代码大模型_CodeAgent与SWEbench面试.md](../23_代码大模型_CodeAgent与SWEbench面试.md)，再用本文件做口语复述。

## 381. 代码大模型和普通 LLM 有什么不同？

代码大模型更重视语法、缩进、符号、API、依赖关系、测试反馈和仓库上下文。普通 LLM 主要生成自然语言，代码模型要生成可执行、可编译、可测试的程序。

面试可以说：代码模型的优势是任务有外部验证信号，比如编译器、单测、lint、类型检查和 CI，所以可以把生成、执行、诊断、修复做成闭环。

## 382. 代码补全、代码生成、代码修复有什么区别？

代码补全输入是当前文件前后缀，强调低延迟、风格一致和语法正确。代码生成输入是自然语言需求或函数签名，强调算法实现和边界条件。代码修复输入是 issue、错误日志、仓库和测试失败，输出是 patch，强调定位和验证闭环。

一句话：补全看局部上下文，生成看需求到代码，修复看仓库理解和测试反馈。

## 383. 代码模型训练数据有什么特殊？

代码数据有语法结构、依赖关系、许可证、重复仓库、issue/PR/commit、测试和可执行反馈。清洗时要关注去重、许可证过滤、秘钥扫描、PII 清理、生成文件和 vendor 目录过滤、benchmark 污染。

高分点：代码数据不只是 `.py/.java` 文件，commit message、issue、PR diff、docstring、README 和 unit tests 都是重要监督信号。

## 384. 代码 tokenizer 为什么特殊？

代码有缩进、换行、括号、路径、snake_case、camelCase、符号和稀有标识符。切得太碎会增加上下文长度，切得太粗会导致词表大和泛化差。

Python 这类语言里缩进有语义，IDE 补全还要保留前后缀格式，所以代码 tokenizer 要兼顾自然语言和程序语言。

## 385. HumanEval、MBPP、SWE-bench 分别评估什么？

HumanEval 和 MBPP 主要评估函数级代码生成，输入通常是函数签名、docstring 或短题描述，看生成代码能否通过隐藏测试。SWE-bench 评估真实 GitHub issue 修复，需要模型理解仓库、修改代码并通过测试。

所以 HumanEval 高不等于会修真实仓库。SWE-bench 更接近 agentic coding，但也不能完全代表真实业务工程。

## 386. pass@k 怎么理解？

pass@k 表示采样 k 个候选里至少一个通过测试的概率。常见无偏估计是 `1 - C(n-c,k)/C(n,k)`，其中 n 是采样数，c 是正确数。

pass@1 更接近一次生成体验，pass@k 能反映多次采样搜索能力。k 越大分数越高，但推理成本也越高，真实产品不能只看大 k。

## 387. 为什么代码 benchmark 容易数据污染？

因为题目、题解、隐藏测试或相似代码可能出现在 GitHub、博客、教程和训练集中。模型如果见过答案，benchmark 分数会虚高。

治理方法包括时间切分、exact/near dedup、AST/n-gram/embedding 去重、隐藏测试增强、动态新题和私有 eval。报告时要区分公开 benchmark 和业务私有评测。

## 388. EvalPlus、LiveCodeBench、BigCodeBench 各解决什么问题？

EvalPlus 给 HumanEval/MBPP 增加更多测试，减少“碰巧过原测试”的情况。LiveCodeBench 用时间切分和持续新题降低污染，更关注较新的代码题。BigCodeBench 更强调复杂指令、库使用和更接近真实任务的代码生成。

它们都比单看 HumanEval 更稳，但仍不能完全替代仓库级 issue 修复评测。

## 389. 仓库级代码理解怎么做？

先构建 repo tree、README/docs、依赖文件、symbol index、imports/call graph、相关测试和 issue/stack trace。检索时结合关键词、embedding、AST/symbol、调用关系、git history 和测试映射。

不要把整个仓库塞进上下文，而是通过检索和符号分析选出最相关文件、函数、测试和错误日志。

## 390. Code RAG 和普通文档 RAG 有什么区别？

普通文档 RAG 的基本单元是段落和 chunk，目标是回答自然语言问题。Code RAG 的基本单元是文件、函数、类、symbol 和测试，目标可能是解释代码、定位 bug 或生成 patch。

代码 RAG 还要考虑 import、调用图、类型、测试文件、版本和分支。正确性不仅是“有证据”，还要语法、行为和测试都过。

## 391. Code Agent 的最小闭环是什么？

最小闭环是：读 issue，检查仓库，定位相关文件，制定计划，生成 patch，运行测试，读取失败日志，修复 patch，最后输出 diff 和验证证据。

核心不是一次性生成代码，而是让模型利用编译器和测试反馈迭代修复。

## 392. 为什么 Code Agent 要输出 diff/patch？

diff 方便 code review、回滚和 CI，也能限制改动范围。整文件重写容易引入无关格式变化、破坏已有逻辑，还难以审查。

高分点：代码 Agent 应该生成最小 patch，保留公共 API，不改无关文件，不删除测试，不通过硬编码测试答案来作弊。

## 393. 测试闭环怎么设计？

先跑和修改文件相关的局部测试，快速迭代；修完后再跑更大范围的回归测试。除了 unit tests，还可以跑 lint、type check、format、build、integration tests 和 security scan。

每次 patch 都要记录测试命令、退出码和关键日志。不能只让模型自评“应该可以”。

## 394. 代码执行为什么必须沙箱？

模型生成的代码或命令可能删除文件、读取密钥、网络外连、无限循环、消耗资源或执行恶意命令。尤其是 Agent 读取仓库和运行测试时，很容易接触本地环境。

沙箱要限制文件系统、网络、CPU、内存、时间，不放生产密钥，高风险命令要人审，所有执行要有 audit log。

## 395. SWE-bench 为什么更接近真实工程？

SWE-bench 来自真实 GitHub issue，要求模型修改仓库让测试通过。它需要理解 issue、定位文件、修改已有代码、保持接口兼容、运行测试和处理真实依赖。

局限也要讲：环境构建复杂，测试不一定覆盖所有正确性，公开 issue 可能污染，测试通过不等于 patch 完美。因此还需要 code review、安全审查和私有 eval。

## 396. 代码 Agent eval 要看哪些指标？

看 issue resolved rate、test pass rate、patch apply success rate、regression rate、average iterations、time to fix、changed lines/files、human review acceptance、security violation rate、build success rate 和 cost per issue。

轨迹上还要看是否找到正确文件、是否运行正确测试、是否理解失败日志、是否改无关文件、是否补测试、是否执行危险命令。

## 397. Fill-in-the-middle 为什么适合代码补全？

IDE 场景里光标后面通常还有 suffix。只看 prefix 续写可能和后续代码冲突。FIM 把任务变成 `prefix + hole + suffix -> middle code`，能利用前后文生成更符合上下文的补全。

它特别适合中间插入代码、补全函数体和局部修改。

## 398. 代码模型如何做仓库级上下文选择？

常见做法是先解析文件树和依赖，再用关键词、embedding、symbol index、import/call graph、错误栈和测试映射找相关文件。然后把 issue、相关函数、调用方/被调方、测试和必要文档放进上下文。

关键是 context budget。上下文越长不一定越好，噪声会干扰模型，还会增加成本和延迟。

## 399. Code Agent 常见失败有哪些？

常见失败包括定位错文件、误解 issue、生成不适配项目风格的 patch、只修表面测试、改坏无关逻辑、无法构建环境、读不懂测试失败、引入安全漏洞、执行危险命令和无限迭代。

治理方法是 repo retrieval、最小 patch、测试闭环、最大迭代次数、沙箱、安全扫描、人工 review 和 bad case 回流 eval。

## 400. 代码大模型 / Code Agent 项目怎么讲 8 分钟？

按背景、架构、关键设计、指标、难点和结果讲。

背景：根据 issue 或测试失败自动定位代码、生成 patch 并验证。架构：Issue -> Repo Index -> File/Symbol Retrieval -> Planner -> Patch Generator -> Test Runner -> Failure Analyzer -> Repair Loop -> Final Diff。

关键设计：建 repo tree、symbol index、依赖关系和测试映射；结合关键词、embedding、AST 搜索；生成最小 diff；沙箱跑测试、lint、type check；根据失败日志迭代；记录 trace 和最终证据。指标看 issue resolved、test pass、patch apply、平均迭代、review 通过率、成本和安全违规。
