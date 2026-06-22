import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = dirname(here);

const desiredOrder = [
  'README.md',
  '00_两个月冲刺总路线.md',
  '01_资源优先级与口碑来源.md',
  '02_每日学习法.md',
  '长期完善目标.md',
  '03_高频题单100题.md',
  '05_高频题单100题_答案索引.md',
  '06_公式与速查卡.md',
  '08_岗位模拟面试套卷.md',
  '09_公司面经雷达与JD关键词.md',
  '10_简历项目打磨与STAR话术.md',
  '11_大模型系统设计面试.md',
  '12_VLM多模态面试.md',
  '13_LLM安全评测与红队.md',
  '14_端侧小模型与模型压缩面试.md',
  '15_大模型数据工程与数据集构建面试.md',
  '16_大模型评测与实验设计面试.md',
  '17_大模型训练系统与分布式训练面试.md',
  '18_DeepSeek_MoE_MLA与ReasoningModel面试.md',
  '19_长上下文_ContextEngineering与GraphRAG面试.md',
  '20_LLMOps模型网关与可观测性面试.md',
  '21_Agent工程化_ToolCalling与MCP面试.md',
  '22_Embedding_Reranker与向量检索面试.md',
  '23_代码大模型_CodeAgent与SWEbench面试.md',
  '24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md',
  '25_GPU_CUDA_Triton与FlashAttention面试.md',
  '26_开源模型生态_模型选型与ChatTemplate面试.md',
  '27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md',
  '28_ML_DL数学基础_损失函数优化器与指标面试.md',
  '29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md',
  '30_Diffusion_DiT_文生图视频生成与可控生成面试.md',
  '31_SpeechAudioLLM_ASR_TTS与实时语音Agent面试.md',
  '32_搜索推荐广告_LLM_Ranking与LTR面试.md',
  '33_PyTorch训练工程_Autograd_DataLoader_AMP_DDP面试.md',
  '34_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets面试.md',
  '35_Tokenizer_BPE_SentencePiece与Token预算面试.md',
  '36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md',
  '37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md',
  '38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md',
  '39_分布式推理_PD分离_KVCache传输与MoEServing面试.md',
  '40_VLM进阶_高分辨率OCR_Grounding_VideoAgent面试.md',
  '41_长上下文进阶_RoPEScaling_YaRN_LongRoPE_RingAttention面试.md',
  '42_数据合成_数据治理_数据配比与污染检测面试.md',
  'modules/01_Transformer与Attention_完整学习章.md',
  'modules/02_训练微调对齐_完整学习章.md',
  'modules/03_RAG_Agent_MCP_完整学习章.md',
  'modules/04_推理部署与性能优化_完整学习章.md',
  'modules/05_手撕代码_完整学习章.md',
  'modules/06_项目面经与冲刺_完整学习章.md',
  'modules/07_ML_DL_NLP公共基础_完整学习章.md',
  'modules/08_2025_2026大模型热点_完整学习章.md',
  'answers/01_Transformer与结构_答案版.md',
  'answers/02_训练微调对齐_答案版.md',
  'answers/03_RAG与Agent_答案版.md',
  'answers/04_推理部署项目手撕_答案版.md',
  'answers/05_2026补充追问题_答案版.md',
  'answers/06_系统设计追问题_答案版.md',
  'answers/07_VLM多模态追问题_答案版.md',
  'answers/08_LLM安全评测与红队_答案版.md',
  'answers/09_端侧小模型与模型压缩_答案版.md',
  'answers/10_大模型数据工程与数据集构建_答案版.md',
  'answers/11_大模型评测与实验设计_答案版.md',
  'answers/12_大模型训练系统与分布式训练_答案版.md',
  'answers/13_DeepSeek_MoE_MLA_ReasoningModel_答案版.md',
  'answers/14_长上下文_ContextEngineering_GraphRAG_答案版.md',
  'answers/15_LLMOps模型网关与可观测性_答案版.md',
  'answers/16_Agent工程化_ToolCalling_MCP_答案版.md',
  'answers/17_Embedding_Reranker_向量检索_答案版.md',
  'answers/18_代码大模型_CodeAgent_SWEbench_答案版.md',
  'answers/19_推理引擎_vLLM_SGLang_TensorRTLLM_答案版.md',
  'answers/20_GPU_CUDA_Triton与FlashAttention_答案版.md',
  'answers/21_开源模型生态_模型选型_ChatTemplate_答案版.md',
  'answers/22_PromptEngineering_结构化输出_ConstrainedDecoding_答案版.md',
  'answers/23_ML_DL数学基础_损失函数优化器指标_答案版.md',
  'answers/24_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling_答案版.md',
  'answers/25_Diffusion_DiT_文生图视频生成与可控生成_答案版.md',
  'answers/26_SpeechAudioLLM_ASR_TTS与实时语音Agent_答案版.md',
  'answers/27_搜索推荐广告_LLM_Ranking与LTR_答案版.md',
  'answers/28_PyTorch训练工程_Autograd_DataLoader_AMP_DDP_答案版.md',
  'answers/29_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets_答案版.md',
  'answers/30_Tokenizer_BPE_SentencePiece与Token预算_答案版.md',
  'answers/31_解码策略_Sampling_BeamSearch_LogitsProcessor_答案版.md',
  'answers/32_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant_答案版.md',
  'answers/33_推测解码_DraftModel_EAGLE_Medusa_MTP_答案版.md',
  'answers/34_分布式推理_PD分离_KVCache传输与MoEServing_答案版.md',
  'answers/35_VLM进阶_高分辨率OCR_Grounding_VideoAgent_答案版.md',
  'answers/36_长上下文进阶_RoPEScaling_YaRN_LongRoPE_RingAttention_答案版.md',
  'answers/37_数据合成_数据治理_数据配比与污染检测_答案版.md',
  'chapters/01_Transformer与Attention.md',
  'deepdives/01_Attention_RoPE_KVCache_深挖.md',
  'chapters/02_训练微调对齐.md',
  'deepdives/02_LoRA_DPO_GRPO_深挖.md',
  'chapters/03_RAG知识库与检索增强.md',
  'chapters/04_Agent工具调用与MCP.md',
  'deepdives/03_RAG_Agent_MCP_深挖.md',
  'chapters/05_推理部署与性能优化.md',
  'chapters/06_手撕代码训练营.md',
  'chapters/07_项目复盘与简历话术.md',
  'chapters/08_面经追踪与模拟面试.md',
  '04_原始资料下载清单.md',
  '07_外部资料本地索引.md',
  '更新日志.md',
  'templates/八股答案卡片模板.md',
  'templates/项目复盘模板.md',
  'templates/面经复盘模板.md',
  'templates/每日计划模板.md'
];

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    if (name === 'tools' || name === '外部资料_GitHub') continue;
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) out.push(...walk(full));
    if (st.isFile() && name.endsWith('.md')) out.push(full);
  }
  return out;
}

function normalizePath(path) {
  return path.split(sep).join('/');
}

function titleFromMarkdown(markdown, fallback) {
  const h1 = markdown.match(/^#\s+(.+)$/m);
  return h1 ? h1[1].trim() : fallback.replace(/\.md$/, '');
}

function groupFor(path) {
  if (path.startsWith('modules/')) return '完整学习章';
  if (path.startsWith('answers/')) return '答案版';
  if (path.startsWith('chapters/')) return '专题章节';
  if (path.startsWith('deepdives/')) return '理论深挖';
  if (path.startsWith('templates/')) return '模板';
  if (path === 'README.md') return '入口';
  if (path.includes('更新日志')) return '维护';
  if (path.includes('下载清单')) return '维护';
  return '路线与题单';
}

const orderMap = new Map(desiredOrder.map((path, index) => [path, index]));
const docs = walk(root)
  .map((full) => {
    const path = normalizePath(relative(root, full));
    const markdown = readFileSync(full, 'utf8');
    return {
      path,
      title: titleFromMarkdown(markdown, path),
      group: groupFor(path),
      markdown
    };
  })
  .sort((a, b) => {
    const ao = orderMap.has(a.path) ? orderMap.get(a.path) : 999;
    const bo = orderMap.has(b.path) ? orderMap.get(b.path) : 999;
    return ao - bo || a.path.localeCompare(b.path, 'zh-Hans-CN');
  });

const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM学习</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f3;
      --panel: #f9faf5;
      --paper: #fffefa;
      --ink: #232620;
      --muted: #64695f;
      --quiet: #858c7f;
      --line: #d8ded0;
      --strong-line: #bbc8b7;
      --accent: #256f53;
      --accent-2: #6b542a;
      --accent-soft: #e7f1e9;
      --accent-wash: #f1f6ef;
      --code-bg: #eef1e8;
      --mark: #fff1a8;
      --shadow: 0 18px 44px rgba(54, 69, 48, 0.08);
      --radius: 8px;
      --sidebar: 300px;
      --rightbar: 236px;
      --content: 920px;
    }
    * { box-sizing: border-box; }
    html {
      scroll-behavior: smooth;
      overflow-x: hidden;
    }
    body {
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(37,111,83,0.055), transparent 24rem),
        linear-gradient(180deg, rgba(107,84,42,0.045), transparent 18rem),
        var(--bg);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.72;
      text-rendering: optimizeLegibility;
      overflow-x: hidden;
    }
    a {
      color: var(--accent);
      text-decoration-thickness: 1px;
      text-underline-offset: 3px;
      overflow-wrap: anywhere;
    }
    button, input { font: inherit; }
    .app {
      min-height: 100vh;
    }
    .sidebar {
      position: fixed;
      top: 0;
      bottom: 0;
      left: 0;
      width: var(--sidebar);
      height: auto;
      overflow: auto;
      padding: 18px 12px 28px;
      border-right: 1px solid var(--line);
      background: rgba(249,250,245,0.94);
      backdrop-filter: blur(10px);
      scrollbar-gutter: stable;
    }
    .brand {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin: 0 4px 16px;
    }
    .brand h1 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
      line-height: 1.15;
    }
    .brand small {
      color: var(--quiet);
      white-space: nowrap;
      font-size: 12px;
    }
    .search {
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 0 0 12px;
      background: linear-gradient(var(--panel) 78%, rgba(249,250,245,0));
    }
    .search input {
      width: 100%;
      min-height: 40px;
      padding: 9px 11px;
      border: 1px solid var(--strong-line);
      border-radius: var(--radius);
      background: var(--paper);
      color: var(--ink);
      outline: none;
      font-size: 14px;
    }
    .search input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(37,111,83,0.14);
    }
    .group-title {
      margin: 18px 8px 7px;
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0;
    }
    .nav-meta {
      margin: 2px 8px 10px;
      color: var(--quiet);
      font-size: 12px;
    }
    .nav-group {
      margin: 0;
      padding: 7px 0;
      border-top: 1px solid color-mix(in srgb, var(--line), transparent 28%);
    }
    .nav-group:first-of-type { border-top: 0; }
    .nav-group summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 7px 8px;
      border-radius: 7px;
      color: var(--accent-2);
      cursor: pointer;
      font-size: 12px;
      font-weight: 800;
      list-style: none;
      user-select: none;
    }
    .nav-group summary::-webkit-details-marker { display: none; }
    .nav-group summary:hover { background: var(--accent-wash); }
    .nav-group summary::before {
      content: ">";
      display: inline-block;
      color: var(--quiet);
      font-size: 11px;
      transition: transform 160ms ease-out;
    }
    .nav-group[open] summary::before { transform: rotate(90deg); }
    .nav-group-label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
      flex: 1;
    }
    .nav-count {
      color: var(--quiet);
      font-size: 11px;
      font-weight: 650;
    }
    .empty-state {
      margin: 14px 8px;
      padding: 12px;
      border: 1px dashed var(--strong-line);
      border-radius: 8px;
      color: var(--muted);
      font-size: 13px;
      background: var(--accent-wash);
    }
    .doc-list {
      display: grid;
      gap: 3px;
      margin: 3px 0 0;
      padding: 0;
      list-style: none;
    }
    .doc-link {
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
      width: 100%;
      border: 0;
      text-align: left;
      padding: 7px 9px 7px 12px;
      border-radius: 7px;
      background: transparent;
      color: var(--ink);
      cursor: pointer;
      font-size: 13px;
      line-height: 1.34;
      overflow: hidden;
    }
    .doc-link:hover { background: #edf2e9; }
    .doc-link:focus-visible {
      outline: 2px solid rgba(37,111,83,0.45);
      outline-offset: 2px;
    }
    .doc-link.active {
      color: #123f2d;
      background: var(--accent-soft);
      font-weight: 760;
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .main {
      min-width: 0;
      margin-left: var(--sidebar);
      padding: 24px clamp(22px, 4vw, 64px) 76px;
    }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 3;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin: 0 auto 14px;
      max-width: var(--content);
      padding: 8px 0 10px;
      background: linear-gradient(var(--bg) 72%, rgba(246,247,243,0));
      color: var(--muted);
      font-size: 14px;
    }
    #breadcrumb {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .floating-top {
      position: fixed;
      right: 18px;
      bottom: 24px;
      z-index: 8;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--paper);
      color: var(--ink);
      min-height: 40px;
      padding: 8px 13px;
      cursor: pointer;
      white-space: nowrap;
      box-shadow: 0 12px 28px rgba(36, 40, 34, 0.14);
      opacity: 0;
      pointer-events: none;
      transform: translateY(10px);
      transition: opacity 160ms ease, transform 160ms ease, border-color 160ms ease, color 160ms ease;
    }
    body.show-back-to-top .floating-top {
      opacity: 1;
      pointer-events: auto;
      transform: translateY(0);
    }
    .floating-top:hover,
    .floating-top:focus-visible {
      border-color: var(--accent);
      color: var(--accent);
      outline: none;
      box-shadow: 0 0 0 3px rgba(37,111,83,0.12);
    }
    article {
      max-width: var(--content);
      margin: 0 auto;
      padding: clamp(22px, 3.7vw, 46px) clamp(10px, 1.6vw, 18px) 64px;
      background: transparent;
      border: 0;
      border-radius: 0;
      box-shadow: none;
      overflow-wrap: anywhere;
    }
    article > *:first-child { margin-top: 0; }
    article h1 {
      max-width: 18em;
      font-size: clamp(28px, 2.7vw, 38px);
      line-height: 1.18;
      margin: 0 0 26px;
      letter-spacing: 0;
    }
    article h2 {
      margin: 46px 0 14px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      font-size: clamp(21px, 2vw, 25px);
      line-height: 1.28;
    }
    article h3 {
      margin: 30px 0 10px;
      font-size: 19px;
      line-height: 1.35;
    }
    article h4 {
      margin: 22px 0 8px;
      font-size: 16px;
    }
    article p { margin: 13px 0; }
    article ul, article ol {
      padding-left: 1.35rem;
      margin: 12px 0 18px;
    }
    article li { margin: 6px 0; }
    article li > ul,
    article li > ol { margin: 6px 0 8px; }
    article code {
      padding: 2px 5px;
      border-radius: 5px;
      background: var(--code-bg);
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: 0.93em;
      overflow-wrap: anywhere;
    }
    pre {
      position: relative;
      overflow: auto;
      margin: 18px 0 22px;
      padding: 18px 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #22271f;
      color: #f6f3e8;
      line-height: 1.55;
      scrollbar-gutter: stable;
    }
    pre code {
      padding: 0;
      background: transparent;
      color: inherit;
      font-size: 13px;
    }
    .copy-code {
      position: absolute;
      right: 8px;
      top: 8px;
      border: 1px solid rgba(255,255,255,0.2);
      background: rgba(255,255,255,0.08);
      color: #f7f0df;
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 12px;
      cursor: pointer;
    }
    blockquote {
      margin: 20px 0;
      padding: 12px 16px;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      color: #244536;
    }
    .table-scroll {
      width: 100%;
      margin: 20px 0 24px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
    }
    .table-scroll-inner {
      display: block;
      width: 100%;
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      scrollbar-gutter: stable;
    }
    table {
      width: 100%;
      min-width: 680px;
      border-collapse: collapse;
      margin: 0;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      vertical-align: top;
      min-width: 120px;
      font-size: 14px;
      line-height: 1.58;
    }
    th {
      background: #eef3ea;
      font-weight: 760;
    }
    tr:last-child td { border-bottom: 0; }
    th:last-child, td:last-child { border-right: 0; }
    hr {
      border: 0;
      border-top: 1px solid var(--line);
      margin: 34px 0;
    }
    mark { background: var(--mark); padding: 1px 2px; border-radius: 3px; }
    .home-page {
      padding-top: clamp(10px, 2vw, 22px);
    }
    .home-hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: clamp(18px, 3vw, 34px);
      align-items: end;
      padding-bottom: 26px;
      border-bottom: 1px solid var(--line);
    }
    .home-kicker {
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0;
    }
    .home-page .home-title {
      max-width: 13em;
      margin: 0;
      font-size: clamp(34px, 4vw, 56px);
      line-height: 1.04;
    }
    .home-summary {
      max-width: 760px;
      margin: 18px 0 0;
      color: #435047;
      font-size: 16px;
      line-height: 1.78;
    }
    .home-stats {
      display: grid;
      gap: 8px;
      min-width: 174px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
    }
    .home-stat {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      color: var(--muted);
      font-size: 13px;
    }
    .home-stat strong {
      color: var(--ink);
      font-size: 18px;
    }
    .home-quick {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 24px 0 36px;
    }
    .home-quick a,
    .home-card a,
    .home-source a {
      text-decoration: none;
    }
    .quick-link {
      display: block;
      min-height: 92px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--accent-soft);
      color: var(--ink);
    }
    .quick-link strong {
      display: block;
      margin-bottom: 8px;
      font-size: 16px;
    }
    .quick-link span {
      display: block;
      color: #435047;
      font-size: 13px;
      line-height: 1.48;
    }
    .home-section {
      margin: 42px 0 0;
    }
    .home-section-header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    .home-section h2 {
      margin: 0;
      padding: 0;
      border: 0;
      font-size: clamp(22px, 2vw, 28px);
    }
    .home-section-note {
      max-width: 420px;
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      text-align: right;
    }
    .home-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .home-card {
      min-width: 0;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
    }
    .home-card-top {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .home-card h3 {
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
    }
    .home-count {
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 12px;
    }
    .home-card p {
      margin: 0 0 13px;
      color: #47544a;
      font-size: 13px;
      line-height: 1.58;
    }
    .home-links {
      display: grid;
      gap: 8px;
    }
    .home-link {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      min-height: 38px;
      padding: 8px 10px;
      border: 1px solid #e2e7dc;
      border-radius: 7px;
      background: #fbfcf7;
      color: var(--ink);
    }
    .home-link:hover,
    .home-link:focus-visible,
    .quick-link:hover,
    .quick-link:focus-visible {
      border-color: var(--accent);
      outline: none;
      box-shadow: 0 0 0 3px rgba(37,111,83,0.12);
    }
    .home-link-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
      font-weight: 650;
    }
    .home-link-tag {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .home-source {
      margin-top: 34px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #eef3ea;
    }
    .home-source h2 {
      margin: 0 0 10px;
      padding: 0;
      border: 0;
      font-size: 20px;
    }
    .home-source-links {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .home-source-links a {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid #cddbcf;
      border-radius: 7px;
      background: var(--paper);
      color: #123f2d;
      font-size: 13px;
      font-weight: 700;
    }
    .doc-pager {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
      gap: 10px;
      margin-top: 48px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
    }
    .pager-button {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--paper);
      color: var(--ink);
      min-height: 42px;
      padding: 8px 12px;
      cursor: pointer;
      font-size: 14px;
      line-height: 1.25;
      text-align: center;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .pager-button:hover,
    .pager-button:focus-visible {
      border-color: var(--accent);
      color: var(--accent);
      outline: none;
      box-shadow: 0 0 0 3px rgba(37,111,83,0.12);
    }
    .pager-button[disabled] {
      cursor: not-allowed;
      color: #a0a79e;
      background: #f1f2ee;
      box-shadow: none;
    }
    .pager-button.catalog {
      min-width: 86px;
      color: #123f2d;
      font-weight: 700;
      background: var(--accent-soft);
      border-color: #c9d9cd;
    }
    .pager-button.prev { text-align: left; }
    .pager-button.next { text-align: right; }
    .rightbar {
      display: none;
      position: fixed;
      top: 0;
      right: 0;
      bottom: 0;
      width: var(--rightbar);
      height: auto;
      overflow: auto;
      padding: 24px 16px;
      border-left: 1px solid var(--line);
      background: rgba(249,250,245,0.88);
      color: var(--muted);
      scrollbar-gutter: stable;
    }
    .rightbar h2 {
      margin: 0 0 10px;
      color: var(--ink);
      font-size: 14px;
    }
    .toc {
      display: grid;
      gap: 2px;
    }
    .toc a {
      display: block;
      padding: 5px 0 5px 9px;
      border-left: 2px solid transparent;
      color: var(--muted);
      text-decoration: none;
      font-size: 13px;
      line-height: 1.35;
    }
    .toc a:hover {
      color: var(--accent);
      border-left-color: var(--accent);
    }
    .toc a.active {
      color: var(--accent);
      border-left-color: var(--accent);
      background: var(--accent-wash);
      font-weight: 760;
    }
    .toc .h3 { padding-left: 20px; font-size: 12px; }
    .stats {
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      font-size: 13px;
    }
    .mobile-menu {
      display: none;
      position: sticky;
      top: 0;
      z-index: 5;
      padding: 10px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    .mobile-title {
      margin-left: auto;
      color: var(--muted);
      font-size: 13px;
      font-weight: 760;
    }
    .sidebar-backdrop { display: none; }
    @media (min-width: 1480px) {
      .main { margin-right: var(--rightbar); }
      .rightbar { display: block; }
      .floating-top { right: calc(var(--rightbar) + 24px); }
    }
    @media (max-width: 760px) {
      .mobile-menu {
        display: flex;
        gap: 8px;
        align-items: center;
      }
      .mobile-menu button {
        border: 1px solid var(--line);
        border-radius: 7px;
        background: var(--paper);
        min-height: 44px;
        padding: 8px 12px;
      }
      .app { display: block; }
      .sidebar-backdrop {
        display: none;
        position: fixed;
        inset: 53px 0 0;
        z-index: 5;
        background: rgba(35, 38, 32, 0.22);
      }
      body.sidebar-open .sidebar-backdrop { display: block; }
      body.sidebar-open { overflow: hidden; }
      .sidebar {
        display: none;
        position: fixed;
        inset: 53px auto 0 0;
        z-index: 6;
        width: min(92vw, 370px);
        height: auto;
        box-shadow: 18px 0 38px rgba(35,38,32,0.16);
      }
      body.sidebar-open .sidebar { display: block; }
      .main { padding: 10px 10px 48px; }
      .main {
        margin-left: 0;
        margin-right: 0;
      }
      .toolbar {
        position: static;
        margin-bottom: 10px;
        padding: 8px 4px;
        font-size: 13px;
      }
      article {
        padding: 20px 6px 48px;
        box-shadow: none;
      }
      article h1 { font-size: 27px; }
      article h2 { margin-top: 36px; }
      .home-hero {
        grid-template-columns: 1fr;
        align-items: start;
      }
      .home-stats {
        min-width: 0;
      }
      .home-quick,
      .home-grid {
        grid-template-columns: 1fr;
      }
      .home-section-header {
        display: block;
      }
      .home-section-note {
        margin-top: 8px;
        text-align: left;
      }
      .doc-pager {
        grid-template-columns: 1fr;
        gap: 8px;
      }
      .pager-button,
      .pager-button.prev,
      .pager-button.next {
        text-align: center;
      }
      .floating-top {
        right: 14px;
        bottom: 14px;
      }
      table { min-width: 620px; }
    }
  </style>
</head>
<body>
  <div class="mobile-menu">
    <button id="toggleSidebar" type="button" aria-expanded="false">目录</button>
    <button id="focusSearch" type="button">搜索</button>
    <span class="mobile-title">LLM学习</span>
  </div>
  <div id="sidebarBackdrop" class="sidebar-backdrop" aria-hidden="true"></div>
  <div class="app">
    <aside class="sidebar" id="sidebar">
      <div class="brand">
        <h1>LLM学习</h1>
        <small>单页版</small>
      </div>
      <div class="search">
        <input id="searchInput" type="search" placeholder="搜索章节或正文，按 / 聚焦" autocomplete="off">
      </div>
      <div id="docNav"></div>
    </aside>
    <main class="main">
      <div class="toolbar">
        <span id="breadcrumb">准备加载</span>
      </div>
      <article id="content"></article>
    </main>
    <aside class="rightbar">
      <h2>本章目录</h2>
      <nav id="toc" class="toc"></nav>
      <div id="stats" class="stats"></div>
    </aside>
    <button id="topButton" class="floating-top" type="button" aria-label="回到页面顶部">↑ 顶部</button>
  </div>
  <script id="docs-data" type="application/json">${JSON.stringify(docs).replace(/</g, '\\u003c')}</script>
  <script>
    const docs = JSON.parse(document.getElementById('docs-data').textContent);
    const docByPath = new Map(docs.map((doc) => [doc.path, doc]));
    const content = document.getElementById('content');
    const docNav = document.getElementById('docNav');
    const toc = document.getElementById('toc');
    const stats = document.getElementById('stats');
    const breadcrumb = document.getElementById('breadcrumb');
    const searchInput = document.getElementById('searchInput');
    const toggleSidebar = document.getElementById('toggleSidebar');
    const focusSearch = document.getElementById('focusSearch');
    const sidebarBackdrop = document.getElementById('sidebarBackdrop');
    const topButton = document.getElementById('topButton');
    const defaultOpenGroups = new Set(['入口', '路线与题单', '完整学习章']);
    let headingObserver = null;
    let tocHeadings = [];
    let tocScrollScheduled = false;
    let currentPath = docs[0]?.path || '';
    const homeQuickLinks = [
      { path: '00_两个月冲刺总路线.md', label: '先定路线', note: '按目标方向决定先学什么、少学什么。' },
      { path: 'modules/01_Transformer与Attention_完整学习章.md', label: '核心主线', note: '从 Transformer 和 Attention 开始打底。' },
      { path: '03_高频题单100题.md', label: '开始问答', note: '用高频题单快速发现薄弱点。' }
    ];
    const homeSections = [
      {
        title: '基础路线与学习方法',
        note: '适合刚进入资料库时先看，先建立节奏和优先级。',
        items: [
          ['00_两个月冲刺总路线.md', '路线'],
          ['02_每日学习法.md', '方法'],
          ['01_资源优先级与口碑来源.md', '资源'],
          ['长期完善目标.md', '维护'],
          ['06_公式与速查卡.md', '速查'],
          ['05_高频题单100题_答案索引.md', '索引']
        ]
      },
      {
        title: 'Transformer 与模型结构',
        note: '最核心的理论主线：公式、shape、复杂度和手写实现都在这里。',
        items: [
          ['modules/01_Transformer与Attention_完整学习章.md', '完整章'],
          ['chapters/01_Transformer与Attention.md', '专题'],
          ['deepdives/01_Attention_RoPE_KVCache_深挖.md', '深挖'],
          ['answers/01_Transformer与结构_答案版.md', '答案'],
          ['35_Tokenizer_BPE_SentencePiece与Token预算面试.md', 'Tokenizer'],
          ['36_解码策略_Sampling_BeamSearch与LogitsProcessor面试.md', '解码'],
          ['41_长上下文进阶_RoPEScaling_YaRN_LongRoPE_RingAttention面试.md', '长上下文']
        ]
      },
      {
        title: '训练、微调与对齐',
        note: '覆盖 SFT、LoRA、DPO、ORPO、GRPO、HuggingFace 工程和后训练热点。',
        items: [
          ['modules/02_训练微调对齐_完整学习章.md', '完整章'],
          ['chapters/02_训练微调对齐.md', '专题'],
          ['deepdives/02_LoRA_DPO_GRPO_深挖.md', '深挖'],
          ['answers/02_训练微调对齐_答案版.md', '答案'],
          ['29_ReasoningPostTraining_RLVR_Verifier_TestTimeScaling面试.md', 'Reasoning'],
          ['34_HuggingFace微调工程_Transformers_PEFT_TRL_Datasets面试.md', '工程'],
          ['17_大模型训练系统与分布式训练面试.md', '训练系统']
        ]
      },
      {
        title: 'RAG、Agent 与工具调用',
        note: '面向大模型应用落地：检索、重排、工具、MCP、Agent 状态与评估。',
        items: [
          ['modules/03_RAG_Agent_MCP_完整学习章.md', '完整章'],
          ['chapters/03_RAG知识库与检索增强.md', 'RAG'],
          ['chapters/04_Agent工具调用与MCP.md', 'Agent'],
          ['deepdives/03_RAG_Agent_MCP_深挖.md', '深挖'],
          ['21_Agent工程化_ToolCalling与MCP面试.md', '工程'],
          ['22_Embedding_Reranker与向量检索面试.md', '检索'],
          ['answers/03_RAG与Agent_答案版.md', '答案']
        ]
      },
      {
        title: '推理部署、性能与 Infra',
        note: '面向服务落地：延迟、吞吐、KV Cache、量化、推测解码和 GPU 优化。',
        items: [
          ['modules/04_推理部署与性能优化_完整学习章.md', '完整章'],
          ['chapters/05_推理部署与性能优化.md', '专题'],
          ['24_推理引擎_vLLM_SGLang_TensorRTLLM面试.md', '引擎'],
          ['25_GPU_CUDA_Triton与FlashAttention面试.md', 'GPU'],
          ['37_模型量化_低比特推理_GPTQ_AWQ_SmoothQuant面试.md', '量化'],
          ['38_推测解码_DraftModel_EAGLE_Medusa_MTP面试.md', '推测解码'],
          ['39_分布式推理_PD分离_KVCache传输与MoEServing面试.md', '分布式'],
          ['20_LLMOps模型网关与可观测性面试.md', 'LLMOps']
        ]
      },
      {
        title: '手撕代码与工程训练',
        note: '把能讲清楚变成能写出来，适合每天计时练。',
        items: [
          ['modules/05_手撕代码_完整学习章.md', '完整章'],
          ['chapters/06_手撕代码训练营.md', '训练营'],
          ['answers/04_推理部署项目手撕_答案版.md', '答案'],
          ['33_PyTorch训练工程_Autograd_DataLoader_AMP_DDP面试.md', 'PyTorch'],
          ['06_公式与速查卡.md', '速查'],
          ['templates/每日计划模板.md', '模板']
        ]
      },
      {
        title: '项目、系统设计与面试表达',
        note: '把知识变成项目话术、简历证据和模拟面试回答。',
        items: [
          ['modules/06_项目面经与冲刺_完整学习章.md', '完整章'],
          ['chapters/07_项目复盘与简历话术.md', '项目'],
          ['chapters/08_面经追踪与模拟面试.md', '面经'],
          ['08_岗位模拟面试套卷.md', '套卷'],
          ['09_公司面经雷达与JD关键词.md', 'JD'],
          ['10_简历项目打磨与STAR话术.md', '简历'],
          ['11_大模型系统设计面试.md', '系统设计'],
          ['answers/06_系统设计追问题_答案版.md', '答案']
        ]
      },
      {
        title: '多模态、语音、安全与 AIGC',
        note: '覆盖 VLM、OCR、视频、语音、端侧、安全和生成式视觉方向。',
        items: [
          ['12_VLM多模态面试.md', 'VLM'],
          ['40_VLM进阶_高分辨率OCR_Grounding_VideoAgent面试.md', 'VLM 进阶'],
          ['30_Diffusion_DiT_文生图视频生成与可控生成面试.md', 'AIGC'],
          ['31_SpeechAudioLLM_ASR_TTS与实时语音Agent面试.md', '语音'],
          ['13_LLM安全评测与红队.md', '安全'],
          ['14_端侧小模型与模型压缩面试.md', '端侧'],
          ['answers/07_VLM多模态追问题_答案版.md', '答案'],
          ['answers/08_LLM安全评测与红队_答案版.md', '答案']
        ]
      },
      {
        title: '数据、评测与公共基础',
        note: '把实验、指标、数据治理和 ML/DL/NLP 基础补扎实。',
        items: [
          ['modules/07_ML_DL_NLP公共基础_完整学习章.md', '完整章'],
          ['28_ML_DL数学基础_损失函数优化器与指标面试.md', '基础'],
          ['15_大模型数据工程与数据集构建面试.md', '数据'],
          ['16_大模型评测与实验设计面试.md', '评测'],
          ['42_数据合成_数据治理_数据配比与污染检测面试.md', '治理'],
          ['32_搜索推荐广告_LLM_Ranking与LTR面试.md', '搜广推'],
          ['answers/23_ML_DL数学基础_损失函数优化器指标_答案版.md', '答案'],
          ['answers/37_数据合成_数据治理_数据配比与污染检测_答案版.md', '答案']
        ]
      },
      {
        title: '开源生态、模板与外部资料',
        note: '适合查链接、看来源、复盘项目和继续扩展资料库。',
        items: [
          ['26_开源模型生态_模型选型与ChatTemplate面试.md', '生态'],
          ['27_PromptEngineering_结构化输出与ConstrainedDecoding面试.md', 'Prompt'],
          ['04_原始资料下载清单.md', '下载'],
          ['07_外部资料本地索引.md', '索引'],
          ['templates/八股答案卡片模板.md', '模板'],
          ['templates/项目复盘模板.md', '模板'],
          ['templates/面经复盘模板.md', '模板'],
          ['更新日志.md', '日志']
        ]
      }
    ];

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function escapeAttr(value) {
      return escapeHtml(value).replace(/\\n/g, ' ');
    }

    function slugify(value) {
      return String(value)
        .trim()
        .toLowerCase()
        .replace(/<[^>]+>/g, '')
        .replace(/[\\s\\/\\\\]+/g, '-')
        .replace(/[^\u4e00-\u9fa5a-z0-9_-]/g, '')
        .replace(/-+/g, '-')
        .replace(/^-|-$/g, '') || 'section';
    }

    function resolveDocLink(href, fromPath) {
      const raw = href.split('#')[0];
      const hash = href.includes('#') ? '#' + href.split('#').slice(1).join('#') : '';
      if (!raw.endsWith('.md')) return { path: raw, hash, isDoc: false };
      const baseParts = fromPath.split('/');
      baseParts.pop();
      for (const part of raw.split('/')) {
        if (!part || part === '.') continue;
        if (part === '..') baseParts.pop();
        else baseParts.push(part);
      }
      return { path: baseParts.join('/'), hash, isDoc: true };
    }

    function inlineMarkdown(text, fromPath) {
      const codeSpans = [];
      let escaped = String(text).replace(new RegExp('\\\\x60([^\\\\x60]+)\\\\x60', 'g'), (_, code) => {
        const token = '§CODE' + codeSpans.length + '§';
        codeSpans.push('<code>' + escapeHtml(code) + '</code>');
        return token;
      });
      escaped = escapeHtml(escaped);
      escaped = escaped.replace(/\\[([^\\]]+)\\]\\(([^\\)]+)\\)/g, (_, label, href) => {
        const cleanHref = href.trim();
        const resolved = resolveDocLink(cleanHref, fromPath);
        const text = label;
        if (resolved.isDoc && docByPath.has(resolved.path)) {
          return '<a href="#doc=' + encodeURIComponent(resolved.path) + resolved.hash + '" data-doc="' + escapeAttr(resolved.path) + '">' + text + '</a>';
        }
        return '<a href="' + escapeAttr(cleanHref) + '" target="' + (cleanHref.startsWith('http') ? '_blank' : '_self') + '" rel="noreferrer">' + text + '</a>';
      });
      escaped = escaped.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
      escaped = escaped.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
      codeSpans.forEach((html, index) => {
        escaped = escaped.replace('§CODE' + index + '§', html);
      });
      return escaped;
    }

    function renderTable(lines, fromPath) {
      const cells = (line) => line.trim().replace(/^\\|/, '').replace(/\\|$/, '').split('|').map((cell) => cell.trim());
      const header = cells(lines[0]);
      const rows = lines.slice(2).map(cells);
      return '<div class="table-scroll"><div class="table-scroll-inner"><table><thead><tr>' + header.map((cell) => '<th>' + inlineMarkdown(cell, fromPath) + '</th>').join('') +
        '</tr></thead><tbody>' + rows.map((row) => '<tr>' + row.map((cell) => '<td>' + inlineMarkdown(cell, fromPath) + '</td>').join('') + '</tr>').join('') +
        '</tbody></table></div></div>';
    }

    function renderMarkdown(markdown, fromPath) {
      const lines = markdown.replace(/\\r\\n/g, '\\n').split('\\n');
      const out = [];
      let i = 0;
      const headingCounts = new Map();

      while (i < lines.length) {
        const line = lines[i];
        if (!line.trim()) { i += 1; continue; }

        const fence = line.match(new RegExp('^\\\\x60\\\\x60\\\\x60\\\\s*([^\\\\x60]*)$'));
        if (fence) {
          const lang = fence[1].trim();
          i += 1;
          const code = [];
          while (i < lines.length && !lines[i].startsWith('\\x60\\x60\\x60')) {
            code.push(lines[i]);
            i += 1;
          }
          if (i < lines.length) i += 1;
          out.push('<pre><button class="copy-code" type="button">复制</button><code class="language-' + escapeAttr(lang) + '">' + escapeHtml(code.join('\\n')) + '</code></pre>');
          continue;
        }

        const heading = line.match(/^(#{1,6})\\s+(.+)$/);
        if (heading) {
          const level = heading[1].length;
          const text = inlineMarkdown(heading[2].trim(), fromPath);
          let id = slugify(heading[2]);
          const count = headingCounts.get(id) || 0;
          headingCounts.set(id, count + 1);
          if (count) id += '-' + count;
          out.push('<h' + level + ' id="' + escapeAttr(id) + '">' + text + '</h' + level + '>');
          i += 1;
          continue;
        }

        if (/^---+$/.test(line.trim())) {
          out.push('<hr>');
          i += 1;
          continue;
        }

        if (line.trim().startsWith('>')) {
          const quote = [];
          while (i < lines.length && lines[i].trim().startsWith('>')) {
            quote.push(lines[i].replace(/^\\s*>\\s?/, ''));
            i += 1;
          }
          out.push('<blockquote>' + quote.map((part) => '<p>' + inlineMarkdown(part, fromPath) + '</p>').join('') + '</blockquote>');
          continue;
        }

        if (line.includes('|') && i + 1 < lines.length && /^\\s*\\|?\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?\\s*$/.test(lines[i + 1])) {
          const tableLines = [lines[i], lines[i + 1]];
          i += 2;
          while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
            tableLines.push(lines[i]);
            i += 1;
          }
          out.push(renderTable(tableLines, fromPath));
          continue;
        }

        const unordered = line.match(/^\\s*[-*]\\s+(.+)$/);
        const ordered = line.match(/^\\s*\\d+[.)]\\s+(.+)$/);
        if (unordered || ordered) {
          const tag = unordered ? 'ul' : 'ol';
          const items = [];
          while (i < lines.length) {
            const m = tag === 'ul' ? lines[i].match(/^\\s*[-*]\\s+(.+)$/) : lines[i].match(/^\\s*\\d+[.)]\\s+(.+)$/);
            if (!m) break;
            let body = m[1];
            const box = body.match(/^\\[( |x|X)\\]\\s+(.+)$/);
            if (box) {
              const checked = box[1].toLowerCase() === 'x' ? ' checked' : '';
              body = '<input type="checkbox" disabled' + checked + '> ' + inlineMarkdown(box[2], fromPath);
            } else {
              body = inlineMarkdown(body, fromPath);
            }
            items.push('<li>' + body + '</li>');
            i += 1;
          }
          out.push('<' + tag + '>' + items.join('') + '</' + tag + '>');
          continue;
        }

        const para = [line.trim()];
        i += 1;
        while (i < lines.length && lines[i].trim() && !/^(#{1,6})\\s+/.test(lines[i]) && !lines[i].startsWith('\\x60\\x60\\x60') && !/^\\s*[-*]\\s+/.test(lines[i]) && !/^\\s*\\d+[.)]\\s+/.test(lines[i])) {
          if (lines[i].includes('|') && i + 1 < lines.length && /^\\s*\\|?\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?\\s*$/.test(lines[i + 1])) break;
          para.push(lines[i].trim());
          i += 1;
        }
        out.push('<p>' + inlineMarkdown(para.join(' '), fromPath) + '</p>');
      }

      return out.join('\\n');
    }

    function groupDocs(filteredDocs) {
      const groups = new Map();
      for (const doc of filteredDocs) {
        if (!groups.has(doc.group)) groups.set(doc.group, []);
        groups.get(doc.group).push(doc);
      }
      return groups;
    }

    function renderNav() {
      const q = searchInput.value.trim().toLowerCase();
      const filtered = q
        ? docs.filter((doc) => (doc.title + ' ' + doc.path + ' ' + doc.markdown).toLowerCase().includes(q))
        : docs;
      const groups = groupDocs(filtered);
      if (!filtered.length) {
        docNav.innerHTML = '<div class="nav-meta">0 / ' + docs.length + ' 个文档</div><p class="empty-state">没有匹配内容</p>';
        return;
      }
      docNav.innerHTML = '<div class="nav-meta">' + filtered.length + ' / ' + docs.length + ' 个文档</div>' + Array.from(groups.entries()).map(([group, groupDocs]) => {
        const hasCurrent = groupDocs.some((doc) => doc.path === currentPath);
        const open = q || hasCurrent || defaultOpenGroups.has(group) ? ' open' : '';
        return '<details class="nav-group"' + open + '><summary><span class="nav-group-label">' + escapeHtml(group) + '</span><span class="nav-count">' + groupDocs.length + '</span></summary><ul class="doc-list">' +
          groupDocs.map((doc) => '<li><button class="doc-link' + (doc.path === currentPath ? ' active' : '') + '" type="button" data-path="' + escapeAttr(doc.path) + '" title="' + escapeAttr(doc.title + ' · ' + doc.path) + '">' + escapeHtml(doc.title) + '</button></li>').join('') +
          '</ul></details>';
      }).join('');
    }

    function renderToc() {
      const headings = Array.from(content.querySelectorAll('h2, h3')).slice(0, 80);
      tocHeadings = headings;
      if (headingObserver) headingObserver.disconnect();
      toc.innerHTML = headings.map((h) => '<a class="' + h.tagName.toLowerCase() + '" href="#' + encodeURIComponent(h.id) + '" data-heading="' + escapeAttr(h.id) + '">' + escapeHtml(h.textContent) + '</a>').join('') || '<span>本章没有二级目录</span>';
      observeHeadings(headings);
      updateActiveTocFromScroll();
    }

    function setActiveToc(id) {
      toc.querySelectorAll('a').forEach((link) => {
        link.classList.toggle('active', link.dataset.heading === id);
      });
    }

    function observeHeadings(headings) {
      if (!headings.length || !('IntersectionObserver' in window)) return;
      headingObserver = new IntersectionObserver((entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActiveToc(visible[0].target.id);
      }, { rootMargin: '-18% 0px -72% 0px', threshold: 0 });
      headings.forEach((heading) => headingObserver.observe(heading));
    }

    function updateActiveTocFromScroll() {
      if (!tocHeadings.length) return;
      let active = tocHeadings[0];
      for (const heading of tocHeadings) {
        if (heading.getBoundingClientRect().top <= 170) active = heading;
        else break;
      }
      if (active) setActiveToc(active.id);
    }

    function addCopyButtons() {
      content.querySelectorAll('.copy-code').forEach((button) => {
        button.addEventListener('click', async () => {
          const code = button.parentElement.querySelector('code')?.textContent || '';
          try {
            await navigator.clipboard.writeText(code);
            button.textContent = '已复制';
            setTimeout(() => button.textContent = '复制', 1200);
          } catch {
            button.textContent = '复制失败';
            setTimeout(() => button.textContent = '复制', 1200);
          }
        });
      });
    }

    function renderHomeDocLink(path, tag) {
      const doc = docByPath.get(path);
      if (!doc) return '';
      return '<a class="home-link" href="#doc=' + encodeURIComponent(doc.path) + '" data-doc="' + escapeAttr(doc.path) + '">' +
        '<span class="home-link-title">' + escapeHtml(doc.title) + '</span>' +
        '<span class="home-link-tag">' + escapeHtml(tag || doc.group) + '</span>' +
        '</a>';
    }

    function renderHomeCard(section) {
      const links = section.items.map(([path, tag]) => renderHomeDocLink(path, tag)).filter(Boolean);
      if (!links.length) return '';
      return '<section class="home-card">' +
        '<div class="home-card-top"><h3>' + escapeHtml(section.title) + '</h3><span class="home-count">' + links.length + ' 个入口</span></div>' +
        '<p>' + escapeHtml(section.note) + '</p>' +
        '<div class="home-links">' + links.join('') + '</div>' +
        '</section>';
    }

    function renderHomePage() {
      const quick = homeQuickLinks.map((item) => {
        const doc = docByPath.get(item.path);
        if (!doc) return '';
        return '<a class="quick-link" href="#doc=' + encodeURIComponent(doc.path) + '" data-doc="' + escapeAttr(doc.path) + '">' +
          '<strong>' + escapeHtml(item.label) + '</strong>' +
          '<span>' + escapeHtml(item.note) + '</span>' +
          '</a>';
      }).filter(Boolean).join('');
      const cards = homeSections.map(renderHomeCard).filter(Boolean).join('');
      return '<div class="home-page">' +
        '<header class="home-hero">' +
          '<div>' +
            '<p class="home-kicker">目录页主页</p>' +
            '<h1 class="home-title">LLM 学习导航</h1>' +
            '<p class="home-summary">按主题整理核心知识、问答答案、工程实践、项目复盘和学习网站入口。先选方向，再进入对应章节；不需要从文件夹里一点点找。</p>' +
          '</div>' +
          '<div class="home-stats" aria-label="资料概览">' +
            '<div class="home-stat"><span>文档</span><strong>' + docs.length + '</strong></div>' +
            '<div class="home-stat"><span>主题块</span><strong>' + homeSections.length + '</strong></div>' +
            '<div class="home-stat"><span>阅读方式</span><strong>单页</strong></div>' +
          '</div>' +
        '</header>' +
        '<section class="home-quick" aria-label="快速开始">' + quick + '</section>' +
        '<section class="home-section">' +
          '<div class="home-section-header">' +
            '<h2>按主题选择学习入口</h2>' +
            '<p class="home-section-note">每个主题块同时放主线章节、专题、深挖和答案版，方便按目标直接跳转。</p>' +
          '</div>' +
          '<div class="home-grid">' + cards + '</div>' +
        '</section>' +
        '<section class="home-source">' +
          '<h2>资料与维护</h2>' +
          '<div class="home-source-links">' +
            renderHomeDocLink('01_资源优先级与口碑来源.md', '来源') +
            renderHomeDocLink('07_外部资料本地索引.md', '索引') +
            renderHomeDocLink('04_原始资料下载清单.md', '下载') +
            renderHomeDocLink('长期完善目标.md', '目标') +
          '</div>' +
        '</section>' +
        '</div>';
    }

    function renderDocPager(path) {
      const index = docs.findIndex((item) => item.path === path);
      const prev = index > 0 ? docs[index - 1] : null;
      const next = index >= 0 && index < docs.length - 1 ? docs[index + 1] : null;
      const catalogPath = docs[0]?.path || path;
      const prevText = prev ? '上一页：' + prev.title : '上一页';
      const nextText = next ? '下一页：' + next.title : '下一页';
      return '<nav class="doc-pager" aria-label="文档翻页">' +
        '<button class="pager-button prev" type="button" data-pager="prev" ' + (prev ? 'data-path="' + escapeAttr(prev.path) + '"' : 'disabled') + '>' + escapeHtml(prevText) + '</button>' +
        '<button class="pager-button catalog" type="button" data-pager="catalog" data-path="' + escapeAttr(catalogPath) + '">目录</button>' +
        '<button class="pager-button next" type="button" data-pager="next" ' + (next ? 'data-path="' + escapeAttr(next.path) + '"' : 'disabled') + '>' + escapeHtml(nextText) + '</button>' +
        '</nav>';
    }

    function setSidebarOpen(open) {
      document.body.classList.toggle('sidebar-open', open);
      toggleSidebar?.setAttribute('aria-expanded', String(open));
    }

    function revealActiveDoc() {
      const active = docNav.querySelector('.doc-link.active');
      if (active) active.scrollIntoView({ block: 'nearest' });
    }

    function updateBackToTop() {
      document.body.classList.toggle('show-back-to-top', window.scrollY > 360);
    }

    function openDoc(path, push = true) {
      const doc = docByPath.get(path) || docs[0];
      if (!doc) return;
      currentPath = doc.path;
      content.innerHTML = (doc.path === 'README.md' ? renderHomePage() : renderMarkdown(doc.markdown, doc.path)) + renderDocPager(doc.path);
      breadcrumb.textContent = doc.group + ' / ' + doc.title;
      stats.innerHTML = '<strong>' + escapeHtml(doc.title) + '</strong><br>' + escapeHtml(doc.path) + '<br>' + doc.markdown.length.toLocaleString('zh-CN') + ' 字符';
      document.title = doc.title + ' - LLM学习';
      renderNav();
      renderToc();
      addCopyButtons();
      requestAnimationFrame(revealActiveDoc);
      if (push) history.replaceState(null, '', '#doc=' + encodeURIComponent(doc.path));
      window.scrollTo({ top: 0, behavior: 'auto' });
      updateBackToTop();
      setSidebarOpen(false);
    }

    docNav.addEventListener('click', (event) => {
      const button = event.target.closest('[data-path]');
      if (!button) return;
      openDoc(button.dataset.path);
    });

    content.addEventListener('click', (event) => {
      const pagerButton = event.target.closest('button[data-pager]');
      if (pagerButton) {
        const path = pagerButton.dataset.path;
        if (path && docByPath.has(path)) openDoc(path);
        return;
      }
      const link = event.target.closest('a[data-doc]');
      if (!link) return;
      event.preventDefault();
      const path = link.dataset.doc;
      openDoc(path);
    });

    toc.addEventListener('click', (event) => {
      const link = event.target.closest('a[data-heading]');
      if (!link) return;
      event.preventDefault();
      const target = document.getElementById(link.dataset.heading);
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        setActiveToc(link.dataset.heading);
      }
    });

    searchInput.addEventListener('input', renderNav);
    topButton?.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
    toggleSidebar?.addEventListener('click', () => setSidebarOpen(!document.body.classList.contains('sidebar-open')));
    sidebarBackdrop?.addEventListener('click', () => setSidebarOpen(false));
    focusSearch?.addEventListener('click', () => {
      setSidebarOpen(true);
      searchInput.focus();
    });
    window.addEventListener('keydown', (event) => {
      if (event.key === '/' && document.activeElement !== searchInput) {
        event.preventDefault();
        if (window.matchMedia('(max-width: 760px)').matches) setSidebarOpen(true);
        searchInput.focus();
      }
      if (event.key === 'Escape') setSidebarOpen(false);
    });
    window.addEventListener('scroll', () => {
      if (tocScrollScheduled) return;
      tocScrollScheduled = true;
      requestAnimationFrame(() => {
        tocScrollScheduled = false;
        updateBackToTop();
        updateActiveTocFromScroll();
      });
    }, { passive: true });
    window.addEventListener('hashchange', () => {
      const next = decodeURIComponent((location.hash.match(/doc=([^#]+)/) || [])[1] || '');
      if (next && next !== currentPath && docByPath.has(next)) openDoc(next, false);
    });

    const initial = decodeURIComponent((location.hash.match(/doc=([^#]+)/) || [])[1] || docs[0]?.path || '');
    openDoc(initial, false);
  </script>
</body>
</html>
`;

writeFileSync(join(root, 'LLM学习.html'), html, 'utf8');
console.log(`Built LLM学习.html with ${docs.length} markdown documents.`);
