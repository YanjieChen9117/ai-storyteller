## 项目实践任务：AI Storyteller（学生版说明）

### 任务理念
通过对提示词工程（Prompt Engineering）与 Agentic 工作流的动手实践，训练你将“故事创意 → 结构化 Story Bible → 页文 → 插画 → 导出”的端到端链路拆解成可迭代、可校验的子步骤，并在每一步注入一致性与质量控制。

### 学习目标（对齐 README）
- 掌握高级提示词工程：把复杂产出拆分为结构化子任务与约束
- 掌握 Agentic 思维与工作流：失败后自动修复、分步验证、可重复执行
- 提升创意可控性与一致性：人物锚点、风格包、调性/叙述者一致
- 形成一套端到端的 AI 故事生产方法论

### 你要完成的“挖空”点位（以 TODO 标注）
本次作业的需要你修改或补全的关键位置如下（均在 `utils.py`）：

1) TODO[Designer]：`build_image_prompt(bible, beat)`
   - 目标：将风格包(style_pack)、角色视觉锚点(visual_anchors)与场景(image_prompt)融合成高质量、可复用的图像提示词。
   - 必做：
     - 从 `bible['art_style']` 提取 `style_tags`/`palette`/`composition_rules`；
     - 汇总并去重 `characters[].visual_anchors`（最多 3 个）；
     - 融合 `beat['image_prompt']`，场景描述控制在 150 字内；
     - 输出一个清晰、可复用的提示词字符串，强调一致性与构图指导。

2) TODO[Author]：`ensure_page_text(...)` 内联注释
   - 目标：扩展文本验证指标与修复循环的“质量信号”（例如阅读等级、重复用词/句式检测）。
   - 建议：
     - 在 `validate_page_text` 中加入额外规则（保持返回结构不变），例如：
       - 句长/词频分布阈值，重复短语检测；
       - 是否包含主要角色名、是否以问号结尾（已实现，可扩展）。
     - 在 `ensure_page_text` 的循环中使用这些指标改写修复提示，促使模型针对性改写。

提示：`ensure_bible/ensure_image` 已给出“生成→校验→失败修复”的基线思路，可对照模仿。

### 高级项（将用时拉升至 2-3 小时）
为体现 Agentic 深化与一致性控制，请在以下点位完成扩展：

3) TODO[Author-Advanced] 文本可读性与重复检测
- 实现 `compute_readability_metrics(text)`：至少提供一个可读性等级（自定义或常见指标近似）。
- 实现 `detect_repetition(text)`：识别重复短语，返回 `repeated_phrases` 与 `repetition_score`。
- 将上述指标并入 `validate_page_text` 或 `ensure_page_text` 的修复回路提示词中。

4) TODO[Designer-Advanced] 图像调色板一致性
- 在 `validate_image(..., check_palette=True, target_palette=...)` 路径下：
  - 使用 `extract_palette_from_image` 获取主色调；
  - 使用 `compare_palette` 计算与 `bible.art_style.palette` 的匹配分；
  - 在严格模式下，当匹配分过低时触发重试；
- 在 UI（`app.py`）中已增加 `enforce_palette` 开关与覆盖样式输入，完成逻辑后即可体验。

5) TODO[Publisher-Advanced] 导出增强
- 在 `export_pdf` 中基于 `cover_title/cover_subtitle/include_toc`：
  - 生成封面页（可读取 `bible.meta.title`、主配色等）；
  - 生成目录页（逐页列出 `page -> summary`）。

### 运行与提交
1. 安装依赖并配置 `.env`：`GEMINI_API_KEY=你的key`
2. 运行测试：
   - `python test_setup.py`（环境/依赖/API连通性）
   - 可选：`python test_imagen.py`（图像生成冒烟）
   - 可选：`python test_export.py`（导出功能）
3. 启动应用：`streamlit run app.py`
4. 在 UI 中：
   - 输入故事创意与页数，点击生成；
   - 观察生成过程与控制台日志；
   - 使用翻书预览与导出。

提交材料建议：
- 你的代码改动（确保 `TODO[Designer]` 与 `TODO[Author]` 已完成）；
- 一段 2-3 分钟的录屏或几张截图，展示你对风格一致性的控制效果；
- 一个简短的说明：你在提示词/规则中加入了哪些设计，为什么。
  - 若实现了高级项，请在说明中勾勒你的检测/评分/修复策略。

### 评分参考（给你自检）
- 必要功能完成度（40%）：`build_image_prompt` 融合到位、文本校验与修复循环有效
- 一致性与可控性（30%）：角色锚点/风格包/构图对输出有稳定影响
- 提示词质量（20%）：清晰、可执行、能引导模型稳定产出
- 工程质量（10%）：代码清晰、变量命名语义化、无明显坏味道

祝你创作愉快！


---

## 逐步操作指南（Step-by-Step）

以下步骤帮助你按部就班完成所有 TODO。建议边做边跑应用观察效果。

### A. 完成 TODO[Designer]：`build_image_prompt`
目标：将风格包 + 角色锚点 + 场景融合成高质量图像提示词。

1. 打开 `utils.py`，定位 `build_style_pack(bible)`，理解返回串的结构（style/palette/composition）。
2. 在 `build_image_prompt(bible, beat)` 内：
   - 读取 `style_pack = build_style_pack(bible)`；
   - 汇总 `anchors = [a for c in bible.get('characters', []) for a in c.get('visual_anchors', [])]`；
   - 去重并截断：`unique = list(dict.fromkeys([a.strip() for a in anchors if a]))[:3]`；
   - 读取场景：`scene = (beat.get('image_prompt') or '').strip()` 并做长度控制：`if len(scene) > 150: scene = scene[:147] + '...'`；
   - 拼接输出：
     ```python
     parts = ["storybook illustration", style_pack]
     if unique: parts.append(f"Characters: {', '.join(unique)}")
     if scene: parts.append(f"Scene: {scene}")
     return ". ".join(p for p in parts if p)
     ```
3. 跑应用（`streamlit run app.py`），生成一次故事，检查图片 prompt 是否包含风格与锚点。

### B. 完成 TODO[Author]：加强文本质量循环
目标：在 `ensure_page_text` 的校验-修复循环中，纳入更多质量信号。

1. 在 `utils.py` 中找到 `validate_page_text`，确认已有指标（句数/词数/是否以问号结尾/是否包含主角/禁用词）。
2. 将下列两个函数实现或完善（若已有骨架）：
   - `compute_readability_metrics(text)`：计算 `avg_words_per_sentence`、`grade_level`（可近似）、`lexical_diversity`；
   - `detect_repetition(text)`：返回 `repetition_score` 与 `repeated_phrases`/`repeated_bigrams`/`repeated_trigrams`。
3. 在 `validate_page_text` 中：
   - 调用上面两个函数，将结果合并入 `metrics`；
   - 在严格模式下增加阈值，例如：
     ```python
     if not (6 <= avg_words_per_sentence <= 25):
         return False, "Average sentence length out of bounds (6-25)", metrics
     if metrics.get('repetition_score', 0) >= 6:
         return False, "Excessive repetition detected", metrics
     ```
4. 在 `ensure_page_text` 中无需改返回结构；若触发失败，将 `reason` 传入 `make_page_text_repair_prompt`，模型会按原因重写。
5. 跑应用，观察文本在第二次尝试后更易达标。

### C. 完成 TODO[Designer-Advanced]：调色板一致性
目标：校验生成图片是否与 `bible.art_style.palette` 接近。

1. 在 `utils.py`：
   - 完成 `extract_palette_from_image(img_bytes, max_colors=5)`（PIL.quantize 提取主色）；
   - 完成 `_hex_to_rgb`（hex/常用色名 → RGB）与 `compare_palette(target_palette, image_palette)`（返回 0-1 分数）。
2. 修改 `validate_image(..., check_palette=True, target_palette=...)`：
   - 提取图片主色，计算分数并写入 `metrics['palette_score']`；
   - 严格模式下 `score < 0.5` 触发失败。
3. 在 UI 勾选 Advanced 的 `Enforce palette consistency`，再次生成故事，观察 metrics 中的 `palette_score`。

### D. 完成 TODO[Publisher-Advanced]：PDF 封面/目录
目标：当传入 `cover_title/cover_subtitle/include_toc` 时，生成封面与目录页。

1. 在 `app.py` 的 `export_pdf` 调用处，临时给函数传参（可在按钮事件里硬编码或从表单读值）。
2. 在 `app.py::export_pdf` 内：
   - 在循环前，如果 `cover_title` 存在：
     - `pdf.add_page()` → 大号标题/副标题/日期/作者（可从 `bible.meta.title` 等推断）；
   - 如果 `include_toc=True`：
     - 添加一页目录：遍历 `pages`，逐行写 `page -> summary`。
3. 生成 PDF，检查第一页是否为封面，第二页是否为目录，后续为正文。

### E. 可选调参建议
- 提高 `ensure_image`/`ensure_page_text` 的 `max_attempts`；
- 将 `strict=True` 用于最终成品，`strict=False` 用于调试；
- 给 `build_image_prompt` 与 `build_page_text_prompt` 添加更多约束（例如镜头语言、角色姿态）。

完成以上步骤，即可达到“按步骤即可完成”的目标。遇到问题，先看控制台/日志，再参考 `TA_instruction.md` 的示例实现片段。
