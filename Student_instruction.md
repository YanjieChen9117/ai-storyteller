## AI Storyteller Project (Student Guide)

### Learning Rationale
This assignment gives you hands-on practice with prompt engineering and agentic workflows. You will build an end-to-end pipeline from “story idea → structured Story Bible → per-page text → illustrations → export,” adding consistency checks and quality control at every stage.

### What You Will Build (End-to-End Logic)
- Architect: Generate a valid Story Bible (structured JSON) with an automatic repair loop on failure.
- Author: Generate page text per beat; use validation metrics to drive a “generate → validate → repair” loop.
- Designer: Compose image prompts that enforce cross-page consistency; optionally validate color palette and retry.
- Publisher: Export images and text to a PDF (optional cover and table of contents).
- UI/Params: Use Advanced toggles and style overrides in the app for rapid iteration and comparisons.

---

## Step-by-Step Guide

### A. Architect: Story Bible Generation & Repair (Understand and Use)
Goal: Ensure `ensure_bible` produces JSON that conforms to the `StoryBible` model; repair when validation fails.

1) Read `utils.py::ensure_bible` and the related `make_bible_prompt_local/make_bible_repair_prompt`.
2) Key points:
   - First attempt uses a strict JSON-only prompt.
   - Validate with `validate_bible`; on failure, send “error + last JSON” for a repair attempt.
   - Return `(StoryBible instance, original dict)` and serialize to `bible.json` in `app.py`.
3) Run the app and verify `bible.json` is consistently produced without console errors.

### B. Designer (Required): Implement `build_image_prompt`
Goal: Merge style pack + character anchors + scene into a high-quality image prompt with consistent look.

Steps:
1) Open `utils.py` and inspect `build_style_pack(bible)` (it returns a concise style/palette/composition string).
2) In `build_image_prompt(bible, beat)`:
   - Set `style_pack = build_style_pack(bible)`.
   - Aggregate and deduplicate character visual anchors (max 3).
   - Read `beat['image_prompt']` and limit to ≤150 characters.
   - Compose the final prompt:
     ```python
     parts = ["storybook illustration", style_pack]
     if anchors: parts.append(f"Characters: {', '.join(anchors)}")
     if scene: parts.append(f"Scene: {scene}")
     return ". ".join(p for p in parts if p)
     ```
3) Run the app and confirm the resulting image prompt contains the expected style and anchors.

### C. Author (Required): Strengthen Text Quality Loop
Goal: Add more quality signals to the `ensure_page_text` validation-repair loop.

Steps:
1) In `utils.py`, implement or refine:
   - `compute_readability_metrics(text)`: include `avg_words_per_sentence`, approximate `grade_level`, `lexical_diversity`, etc.
   - `detect_repetition(text)`: return `repetition_score` and repeated n-grams.
2) Merge these metrics into `validate_page_text` and add thresholds, e.g.:
   ```python
   if not (6 <= avg_words_per_sentence <= 25):
       return False, "Average sentence length out of bounds (6-25)", metrics
   if metrics.get('repetition_score', 0) >= 6:
       return False, "Excessive repetition detected", metrics
   ```
3) Keep the `ensure_page_text` return shape `(text, metrics)` unchanged. On failure, use `make_page_text_repair_prompt` (with `reason`) to guide rewrites.
4) Run the app and observe that a second attempt reaches the target quality more often.

### D. Designer-Advanced (Optional but Recommended): Palette Consistency
Goal: Validate that each image’s dominant colors align with `bible.art_style.palette`, and retry if needed.

Steps:
1) In `utils.py`:
   - Implement `extract_palette_from_image` (e.g., PIL.quantize for dominant colors).
   - Implement `_hex_to_rgb` and `compare_palette` (convert target palette to RGB and compute a 0-1 match score).
2) In `validate_image(..., check_palette=True, target_palette=...)`:
   - Write `metrics['palette_score']`.
   - In strict mode, trigger failure when score < 0.5.
3) In the UI (`app.py`), enable Advanced → “Enforce palette consistency,” regenerate the story, and check improvements.

### E. Publisher-Advanced (Optional): PDF Cover/TOC
Goal: When `cover_title/cover_subtitle/include_toc` are provided, generate a cover and a table of contents.

Steps:
1) In `app.py::export_pdf`, before adding page content:
   - If `cover_title` or `cover_subtitle` exists, add a styled cover page.
   - If `include_toc=True`, add a TOC page listing `page -> summary`.
2) Export and confirm the order: Cover → TOC → Content.

### F. Run & Debug (Checklist)
- `python test_setup.py` passes environment/dependency/API checks.
- `streamlit run app.py` generates `bible.json`, per-page JSON, and images.
- With Advanced toggles ON, logs show `palette_score` and text metrics.
- PDF/ZIP export works.

---

## Submission & Grading

Submit:
- Your code changes (complete B/C as required; D/E are extra credit).
- A short explanation of your prompt/validation design and at least one failure→repair example.

Rubric (self-check):
- Functionality (40%): image prompt fusion; text validation + repair loop works.
- Consistency/Control (30%): anchors/style/composition consistently influence outputs.
- Prompt Quality (20%): clear, executable, and leads to stable outputs.
- Code Quality (10%): clear naming, no obvious code smells, no lint errors.

Good luck and have fun!
