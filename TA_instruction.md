## 项目实践任务：AI Storyteller（助教版答案与评分参考）

本文件提供参考实现要点与评分细则，便于批改作业。允许学生有创造性实现（不必与示例完全一致）。

### 参考答案要点

1) build_image_prompt（位于 `utils.py`）
   - 需融合三部分：
     - style_pack：来自 `bible.art_style` 的 `style_tags`、`palette`、`composition_rules` 的精炼串；
     - anchors：从 `bible.characters[].visual_anchors` 汇总去重，保留前 2~3 个；
     - scene：`beat.image_prompt`，长度控制（<=150 字符）。
   - 推荐输出格式（示例，不拘泥于字面）：
     - `"storybook illustration; style: watercolor, whimsical; palette: warm, bright; composition: storybook framing. Characters: red scarf, round glasses. Scene: ..."`
   - 关键检查点：风格与锚点确实被串入提示词；长度有控制；语义清晰。

2) ensure_page_text（位于 `utils.py`）
   - 在现有“生成→校验→修复”的循环中，扩展 `validate_page_text` 的质量指标即可：
     - 示例指标：平均句长阈值、重复短语检测、是否包含主角名、是否以问号结尾（已有）、禁用词（已有，可扩展）。
   - 修复提示 `make_page_text_repair_prompt` 可注入上述指标的信息，促使模型朝目标改写。
   - 返回结构保持 `(cleaned_text, metrics)` 不变。

### 示例实现（仅供参考）

以下为一种可得满分的参考实现思路，允许学生做等价替代：

```python
# utils.py（片段）
def build_image_prompt(bible: dict, beat: dict) -> str:
    style_pack = build_style_pack(bible)  # 形如："style: watercolor; palette: warm, bright; composition: ..."
    anchors: list[str] = []
    for char in bible.get('characters', []):
        anchors.extend(char.get('visual_anchors', []))
    anchors = list(dict.fromkeys([a.strip() for a in anchors if a.strip()]))[:3]
    anchor_text = f"Characters: {', '.join(anchors)}" if anchors else ""
    scene = (beat.get('image_prompt') or '').strip()
    if len(scene) > 150:
        scene = scene[:147] + '...'
    parts = ["storybook illustration", style_pack]
    if anchor_text:
        parts.append(anchor_text)
    if scene:
        parts.append(f"Scene: {scene}")
    return ". ".join(p for p in parts if p)

def validate_page_text(...):
    # 在现有基础上增加：重复短语、平均句长阈值
    avg_len_ok = 6 <= avg_words_per_sentence <= 25
    if strict and not avg_len_ok:
        return False, "Average sentence length out of bounds", metrics
    # 可添加简单重复检测
    rep = detect_repetition(cleaned_text)
    metrics.update(rep)
    if strict and rep.get('repetition_score', 0) >= 6:
        return False, "Excessive repetition detected", metrics

def validate_image(..., check_palette=False, target_palette=None):
    # 调用 extract_palette_from_image + compare_palette 计算 palette_score
    if check_palette:
        image_palette = extract_palette_from_image(img_bytes)
        metrics['palette_score'] = compare_palette(target_palette or [], image_palette)
        if strict and metrics['palette_score'] < 0.5:
            return False, "Palette mismatch", metrics
```

### 评分细则（建议）

- 功能完成度 40%
  - `build_image_prompt` 三要素融合且输出清晰
  - `ensure_page_text` 循环能因指标失败而触发有意义的修复
- 一致性与可控性 30%
  - 生成结果在不同页保持角色/风格一致；提示词能明显影响产出
- 提示词质量 20%
  - 语言清晰、约束明确、便于模型稳定遵循
- 工程质量 10%
  - 命名清晰、无明显坏味道、无新的 Lint 错误

### 批改建议

- 运行 `streamlit run app.py`，让学生现场演示 1 个故事；观察生成图片与文本是否体现提示词约束。
- 检查 `outputs/<slug>/images` 与 `pages` 是否齐全；翻阅 `bible.json` 是否结构完备。
- 若 API 受限，可接受“占位图”效果，但提示词与校验逻辑仍需到位。

### 快速操作要点（助教）
- 在 UI 的 Advanced 区块开启 palette 检查与样式覆盖，观察图片一致性变化。
- 将 `cover_title/cover_subtitle/include_toc` 传入 `export_pdf` 手动测试导出增强（若学生已实现）。


