"""
app.py — Streamlit single-page app.
Baseline: single-pass pipeline with thin wrappers; students improve prompts/loops.

TODO:
1. Modify the prompt templates below to improve story quality
2. Add agentic loops (e.g., story revision, character consistency checks)
3. Implement better error handling and user feedback
4. Add more creative controls (genre, style, etc.)

To run: streamlit run app.py
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
from streamlit.components.v1 import html as st_html
from fpdf import FPDF

from utils import (
    llm_text,
    gen_image_b64,
    write_text,
    save_bytes,
    StoryBible,
    IMAGE_SIZE,
    llm_json,
    ensure_bible,
    ensure_image,
    ensure_page_text,
)

# ---------- Helpers ----------
def slugify(text: str) -> str:
    import re as _re
    s = _re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return _re.sub(r"-+", "-", s).strip("-") or "story"

def ensure_dirs(base: Path) -> None:
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "pages").mkdir(parents=True, exist_ok=True)

def render_flipbook_spread(pages, base_dir: Path, book_w: int = 1200, book_h: int = 780,
                           author: str | None = None, title: str | None = None):
    """两页展开的翻书效果：左页=整幅插图，右页=文字排版，带书脊阴影与翻页动画。"""
    import base64

    def b64img(p: Path) -> str:
        return base64.b64encode(p.read_bytes()).decode("utf-8")

    page_divs = []
    for p in pages:
        img_path = base_dir / "images" / f"page_{p['page']:02d}.png"
        
        # 调试信息：检查图片文件状态
        if img_path.exists():
            img_b64 = b64img(img_path)
            print(f"✅ 翻书模式：页面 {p['page']} 图片加载成功，大小: {img_path.stat().st_size} bytes")
        else:
            img_b64 = ""
            print(f"❌ 翻书模式：页面 {p['page']} 图片文件不存在: {img_path}")
            # 尝试从其他位置查找图片
            alt_paths = [
                base_dir / f"page_{p['page']:02d}.png",
                base_dir / "images" / f"page_{p['page']:02d}.jpg",
                base_dir / f"page_{p['page']:02d}.jpg"
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    img_b64 = b64img(alt_path)
                    print(f"✅ 翻书模式：页面 {p['page']} 从备用路径加载图片: {alt_path}")
                    break
        
        text_html = (p.get("text") or "").replace("\n", "<br>")
        page_divs.append(f"""
        <div class="page" data-page="{p['page']}" style="display: none;">
          <div class="spread">
            <div class="left-panel">
              {f'<img src="data:image/png;base64,{img_b64}" alt="Page {p["page"]}" class="page-image" />' if img_b64 else f'<div style="background:#f0f0f0;display:flex;align-items:center;justify-content:center;height:100%;color:#666;font-size:14px;border-radius:4px;">图片加载失败<br/>Page {p["page"]}</div>'}
            </div>
            <div class="right-panel">
              <div class="page-title">{(title or "").upper()}</div>
              <div class="page-content">{text_html}</div>
              <div class="page-number">{p['page']}</div>
            </div>
          </div>
        </div>
        """)

    html = f"""
    <style>
      :root {{
        --paper-bg: #fffdf7; --ink: #2b2622; --ink-dim: #6b625a; --accent: #6b4f2c;
        --shadow: rgba(0,0,0,0.3); --page-turn: rgba(0,0,0,0.1);
      }}
      
      .flip-container {{
        width: {book_w}px; height: {book_h + 120}px; margin: 0 auto; position: relative;
        perspective: 1200px; perspective-origin: center;
      }}
      
      .book {{
        width: 100%; height: 100%; position: relative; transform-style: preserve-3d;
        transition: transform 0.8s cubic-bezier(0.645, 0.045, 0.355, 1);
      }}
      
      .page {{
        position: absolute; width: 100%; height: {book_h}px; 
        transform-origin: left center; transition: transform 0.8s cubic-bezier(0.645, 0.045, 0.355, 1);
        backface-visibility: hidden; box-shadow: 0 0 20px var(--shadow);
        border-radius: 8px; overflow: hidden;
      }}
      
      .page.active {{
        display: block !important; z-index: 10;
      }}
      
      .page.flipping {{
        z-index: 20; box-shadow: 0 0 30px var(--page-turn);
      }}
      
      .spread {{
        display: grid; grid-template-columns: 1fr 1fr; height: 100%; position: relative;
        background: var(--paper-bg); border-radius: 8px; overflow: hidden;
      }}
      
      .left-panel, .right-panel {{
        padding: 30px; position: relative; display: flex; flex-direction: column;
      }}
      
      .left-panel {{
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
      }}
      
      .right-panel {{
        background: var(--paper-bg); font-family: 'Georgia', serif; color: var(--ink);
        line-height: 1.8; font-size: 18px; position: relative;
        display: flex; flex-direction: column; height: 100%;
        overflow: hidden; /* 防止内容溢出 */
      }}
      
      .page-image {{
        width: 100%; height: 100%; object-fit: cover; border-radius: 4px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }}
      
      .page-content {{
        flex: 1; overflow-y: auto; padding-right: 10px; text-align: justify;
        font-size: 16px; line-height: 1.6; color: var(--ink);
        word-wrap: break-word; hyphens: auto; 
        padding-bottom: 60px; /* 为页码留出空间 */
        max-height: calc(100% - 80px); /* 确保内容不超出容器 */
      }}
      
      .page-title {{
        font-size: 12px; color: var(--ink-dim); text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 15px; font-weight: 600;
        padding: 8px 0; border-bottom: 1px solid rgba(107, 79, 44, 0.2);
      }}
      
      .page-number {{
        position: absolute; bottom: 15px; right: 20px;
        font-size: 12px; color: var(--ink-dim); font-weight: 500;
        background: rgba(255,255,255,0.8); padding: 4px 8px; border-radius: 4px;
      }}
      
      .controls {{
        text-align: center; margin: 0 auto; padding: 20px; 
        position: absolute; bottom: 20px; left: 0; right: 0;
        background: linear-gradient(to top, rgba(255,255,255,0.95), rgba(255,255,255,0.8));
        backdrop-filter: blur(10px);
        border-radius: 0 0 8px 8px;
      }}
      
      .btn {{
        padding: 12px 24px; margin: 0 8px; border-radius: 25px;
        border: 2px solid var(--accent); background: white; color: var(--accent);
        cursor: pointer; font-weight: 600; transition: all 0.3s ease;
        font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }}
      
      .btn:hover {{
        background: var(--accent); color: white; transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(107, 79, 44, 0.3);
      }}
      
      .btn:disabled {{
        opacity: 0.5; cursor: not-allowed; transform: none;
      }}
      
      .page-indicator {{
        display: inline-block; margin: 0 20px; font-size: 14px;
        color: var(--ink-dim); font-weight: 500;
      }}
      
      .book-spine {{
        position: absolute; left: 50%; top: 0; bottom: 0; width: 4px;
        background: linear-gradient(90deg, rgba(0,0,0,0.1), rgba(0,0,0,0.05) 50%, rgba(0,0,0,0.1));
        transform: translateX(-50%); z-index: 5;
      }}
    </style>

    <div class="flip-container">
      <div class="book-spine"></div>
      <div class="book" id="book">
        {''.join(page_divs)}
      </div>
      
      <div class="controls">
        <button class="btn" id="prevBtn" onclick="previousPage()">⟵ 上一页</button>
        <span class="page-indicator">
          <span id="currentPage">1</span> / <span id="totalPages">{len(pages)}</span>
        </span>
        <button class="btn" id="nextBtn" onclick="nextPage()">下一页 ⟶</button>
      </div>
    </div>

    <script>
      let currentPageIndex = 0;
      const totalPages = {len(pages)};
      const pages = document.querySelectorAll('.page');
      
      function updatePageDisplay() {{
        // 隐藏所有页面
        pages.forEach((page, index) => {{
          page.classList.remove('active');
          page.style.display = 'none';
        }});
        
        // 显示当前页面
        if (pages[currentPageIndex]) {{
          pages[currentPageIndex].classList.add('active');
          pages[currentPageIndex].style.display = 'block';
        }}
        
        // 更新页码显示
        document.getElementById('currentPage').textContent = currentPageIndex + 1;
        
        // 更新按钮状态
        document.getElementById('prevBtn').disabled = currentPageIndex === 0;
        document.getElementById('nextBtn').disabled = currentPageIndex === totalPages - 1;
      }}
      
      function nextPage() {{
        if (currentPageIndex < totalPages - 1) {{
          const currentPage = pages[currentPageIndex];
          currentPage.classList.add('flipping');
          
          setTimeout(() => {{
            currentPageIndex++;
            updatePageDisplay();
            currentPage.classList.remove('flipping');
          }}, 400);
        }}
      }}
      
      function previousPage() {{
        if (currentPageIndex > 0) {{
          const currentPage = pages[currentPageIndex];
          currentPage.classList.add('flipping');
          
          setTimeout(() => {{
            currentPageIndex--;
            updatePageDisplay();
            currentPage.classList.remove('flipping');
          }}, 400);
        }}
      }}
      
      // 键盘导航
      document.addEventListener('keydown', (e) => {{
        if (e.key === 'ArrowRight') nextPage();
        if (e.key === 'ArrowLeft') previousPage();
      }});
      
      // 触摸滑动支持
      let touchStartX = 0;
      let touchEndX = 0;
      
      document.addEventListener('touchstart', (e) => {{
        touchStartX = e.changedTouches[0].screenX;
      }});
      
      document.addEventListener('touchend', (e) => {{
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
      }});
      
      function handleSwipe() {{
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;
        
        if (Math.abs(diff) > swipeThreshold) {{
          if (diff > 0) {{
            nextPage();
          }} else {{
            previousPage();
          }}
        }}
      }}
      
      // 初始化显示
      updatePageDisplay();
    </script>
    """
    
    st_html(html, height=book_h + 150, scrolling=False)

def export_pdf(base_dir: Path, pages: List[Dict[str, Any]], image_size: str = IMAGE_SIZE) -> Path:
    """Export story as PDF with images and text."""
    try:
        w, h = map(int, image_size.split("x"))
        pdf = FPDF(orientation="P", unit="pt", format=(w, h + 200))
        pdf.set_auto_page_break(auto=False)

        for i, page in enumerate(pages, start=1):
            pdf.add_page()
            img_path = base_dir / "images" / f"page_{i:02d}.png"

            # Check if image exists and add it
            if img_path.exists():
                try:
                    pdf.image(str(img_path), x=0, y=0, w=w, h=h)
                except Exception as img_error:
                    print(f"Warning: Could not add image for page {i}: {img_error}")
            else:
                print(f"Warning: Image not found for page {i}: {img_path}")

            # Add text
            pdf.set_xy(36, h + 24)
            pdf.set_font("Helvetica", size=12)
            page_text = page.get("text", "")
            if page_text:
                # Truncate text if too long
                truncated_text = page_text[:1200] if len(page_text) > 1200 else page_text
                pdf.multi_cell(w - 72, 16, txt=truncated_text)
            else:
                pdf.multi_cell(w - 72, 16, txt=f"Page {i} - No text available")

        # Ensure output directory exists
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / "book.pdf"

        # Generate PDF
        pdf.output(str(out_path))

        # Verify file was created
        if not out_path.exists():
            raise Exception(f"PDF file was not created at {out_path}")

        print(f"PDF exported successfully to: {out_path}")
        return out_path

    except Exception as e:
        print(f"Error in export_pdf: {str(e)}")
        raise Exception(f"PDF export failed: {str(e)}")

def export_zip(base_dir: Path) -> Path:
    """Export story files as ZIP archive."""
    try:
        import zipfile

        # Ensure output directory exists
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / "story_export.zip"

        # Check if base_dir has content
        files = list(base_dir.rglob("*"))
        if not files:
            raise Exception(f"No files found in {base_dir}")

        print(f"Found {len(files)} files to zip")

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                if p.is_file():
                    try:
                        arcname = p.relative_to(base_dir)
                        z.write(p, arcname=arcname)
                        print(f"Added to ZIP: {arcname}")
                    except Exception as file_error:
                        print(f"Warning: Could not add file {p} to ZIP: {file_error}")

        # Verify file was created
        if not out_path.exists():
            raise Exception(f"ZIP file was not created at {out_path}")

        print(f"ZIP exported successfully to: {out_path}")
        return out_path

    except Exception as e:
        print(f"Error in export_zip: {str(e)}")
        raise Exception(f"ZIP export failed: {str(e)}")

# ---------- Prompt templates (STUDENTS: Improve these!) ----------
def make_bible_prompt(idea: str, pages: int, schema: dict) -> str:
    """Generate a prompt for creating a Story Bible."""
    return f"""You are The Architect - a master storyteller who creates structured story plans.

Create a Story Bible for: "{idea}"
Target length: {pages} pages

REQUIREMENTS:
1. Output ONLY valid JSON - no markdown, no explanations, no extra text
2. Must exactly match this schema: {json.dumps(schema, ensure_ascii=False)}
3. plot_beats array MUST contain exactly {pages} items
4. Each plot_beat must have: {{ "page": n, "summary": "...(<=20 words)", "image_prompt": "...(<=30 words)" }}
5. Include art_style with: style_tags (array), palette (array), composition_rules (string)

Focus on:
- Clear character development
- Logical plot progression
- Visual consistency
- Age-appropriate themes

Return ONLY the JSON object:"""

def make_page_text_prompt(bible: dict, beat: dict) -> str:
    """
    STUDENT TASK: Improve this prompt to write better page text!
    Consider adding:
    - Dialogue guidelines
    - Pacing controls
    - Vocabulary level adjustments
    - Emotional tone consistency
    """
    return f"""
You are the Author. Write the final page text for this beat.

- Narrator: {bible.get('narrator_voice','third-person, gentle, playful')}
- Tone: {bible.get('tone','warm and imaginative')}
- Audience: {bible.get('meta',{}).get('target_audience','children')}
- Mention key character visual anchors subtly; keep text clear and age-appropriate.
- Return ONLY the page text (no headings or commentary).
- Keep text length appropriate for a single page (2-4 sentences for young children).

BIBLE (short):
title={bible.get('meta',{}).get('title','')}, characters={[c.get('name') for c in bible.get('characters',[])]}

BEAT:
page={beat['page']}, summary={beat['summary']}
""".strip()

def make_image_prompt(bible: dict, beat: dict) -> str:
    """
    STUDENT TASK: Improve this prompt to generate better images!
    Consider adding:
    - Style consistency controls
    - Composition guidelines
    - Color palette enforcement
    - Character pose and expression guidance
    """
    style = ", ".join(bible.get("art_style", {}).get("style_tags", []))
    palette = ", ".join(bible.get("art_style", {}).get("palette", []))
    comp = bible.get("art_style", {}).get("composition_rules", "storybook framing, consistent proportions")
    base = f"storybook illustration, {style}; palette: {palette}; composition: {comp}. Scene: {beat['image_prompt']}"
    anchors = [a for c in bible.get("characters", []) for a in c.get("visual_anchors", [])]
    if anchors and not any(a.lower() in base.lower() for a in anchors):
        base += f" Include: {', '.join(anchors[:2])}."
    return base

# ---------- UI ----------
st.set_page_config(page_title="AI Storyteller", layout="wide")
st.title("Project 1: AI Storyteller")
st.caption("🎨 Bring Your Imagination to Life! Modify prompts and add agentic features based on functions included in utils.py!")

# --- Session defaults (place near top, after title) ---
ss = st.session_state
ss.setdefault("story_ready", False)
ss.setdefault("generated_pages", None)
ss.setdefault("base_dir", None)
ss.setdefault("folder_slug", None)
ss.setdefault("image_size", IMAGE_SIZE)

# export buffers
ss.setdefault("export_pdf_data", None)
ss.setdefault("export_zip_data", None)
ss.setdefault("export_pdf_filename", None)
ss.setdefault("export_zip_filename", None)
ss.setdefault("export_pdf_ready", False)
ss.setdefault("export_zip_ready", False)

# Add student guidance
with st.expander("📚 Student Learning Objectives", expanded=False):
    st.markdown("""
    **Your Role on the Team:**
    - **The Architect**: Engineer prompts to create structured story plans
    - **The Character & Style Designer**: Define the book's unique visual identity  
    - **The Author**: Craft prompts to guide the AI's narrative voice
    - **The Brand Manager**: Give your final project a catchy, creative name!
    
    **What You'll Learn:**
    - Advanced Prompt Engineering
    - Agentic Thinking & Workflows
    - Creative AI Control & Consistency
    """)

# Story controls on main screen
with st.form("controls", clear_on_submit=False):
    st.subheader("Story Setup")
    idea = st.text_input("Core idea", value="a curious kid finds a talking compass", 
                        help="Describe your story concept in one sentence")
    c1, c2 = st.columns(2)
    with c1:
        pages = st.number_input("Target pages", min_value=4, max_value=16, value=8, step=1,
                               help="How many pages should your storybook have?")
    with c2:
        image_size = st.text_input("Image size", value=IMAGE_SIZE, 
                                  help="e.g., 1024x1024 (square) or 1792x1024 (landscape)")
    project_name = st.text_input("Project name", value="", 
                                help="Give your story a creative name!")
    run_button = st.form_submit_button("🚀 Generate Story")

if run_button:
    try:
        # 1) Architect → Bible
        with st.spinner("🎭 Creating Story Bible..."):
            schema = StoryBible.model_json_schema()
            try:
                bible, data = ensure_bible(idea, int(pages), schema, max_attempts=3, temperature=0.4)
            except Exception as json_error:
                error_msg = str(json_error)

                # Handle RetryError by extracting underlying exception
                if "RetryError" in error_msg:
                    try:
                        import re
                        match = re.search(r"Exception: (.+?)(?:\n|$)", error_msg)
                        if match:
                            error_msg = match.group(1)
                    except:
                        pass

                if "authentication failed" in error_msg.lower():
                    st.error(f"❌ API Authentication Error: {error_msg}")
                    st.info("💡 Please check your GEMINI_API_KEY in the .env file. Make sure it's valid and not expired.")
                elif "quota exceeded" in error_msg.lower():
                    st.error(f"❌ API Quota Error: {error_msg}")
                    st.info("💡 You've reached your Gemini API usage limit. Please try again later or check your billing.")
                elif "network connection" in error_msg.lower():
                    st.error(f"❌ Network Error: {error_msg}")
                    st.info("💡 Please check your internet connection and try again.")
                elif "invalid json format" in error_msg.lower() or "json" in error_msg.lower():
                    st.error(f"❌ JSON Format Error: {error_msg}")
                    st.info("💡 The AI generated malformed JSON. Try again or simplify your story idea.")
                else:
                    st.error(f"❌ Unexpected Error: {error_msg}")
                    st.info("💡 An unexpected error occurred. Please try again or contact support if the problem persists.")
                st.stop()

        # Save folder
        folder_slug = slugify(project_name or bible.meta.get("title") or idea)
        base_dir = Path("outputs") / folder_slug
        ensure_dirs(base_dir)

        # Persist bible
        write_text(base_dir / "bible.json", json.dumps(data, indent=2))

        st.subheader("📖 Story Bible")
        st.json(data)

        # 2) Author + 3) Designer
        generated_pages: List[Dict[str, Any]] = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("✍️ Writing pages and generating images..."):
            for i, beat in enumerate(bible.plot_beats[: int(pages) ]):
                status_text.text(f"Creating page {i+1}/{len(bible.plot_beats[:int(pages)])}...")
                progress_bar.progress((i + 1) / len(bible.plot_beats[:int(pages)]))

                beat_dict = beat.model_dump()
                # Author draft (with validation and retry)
                try:
                    page_text, text_metrics = ensure_page_text(
                        data, beat_dict, max_attempts=2, strict=True, temperature=0.7
                    )
                    print(f"Page {beat.page}: Text generated successfully")
                    print(f"  Sentences: {text_metrics.get('sentence_count', 'N/A')}")
                    print(f"  Words: {text_metrics.get('word_count', 'N/A')}")
                    print(f"  Avg words/sentence: {text_metrics.get('avg_words_per_sentence', 'N/A')}")
                    print(f"  Has main character: {text_metrics.get('has_main_character', 'N/A')}")
                except Exception as text_error:
                    print(f"Text generation failed for page {beat.page}: {str(text_error)}")
                    page_text = llm_text(make_page_text_prompt(data, beat_dict), temperature=0.7)

                # Designer prompt + image (with validation and retry)
                try:
                    img_bytes, final_image_prompt, img_metrics = ensure_image(
                        data, beat_dict, image_size, max_attempts=2, strict=False
                    )
                    img_path = base_dir / "images" / f"page_{beat.page:02d}.png"
                    save_bytes(img_path, img_bytes)
                    print(f"Page {beat.page}: Image generated successfully")
                    print(f"  Size: {img_metrics.get('width', 'N/A')}x{img_metrics.get('height', 'N/A')}")
                    print(f"  Colors: {img_metrics.get('unique_colors', 'N/A')}")
                    print(f"  Entropy: {img_metrics.get('entropy', 'N/A')}")
                except Exception as img_error:
                    print(f"Image generation failed for page {beat.page}: {str(img_error)}")
                    final_image_prompt = make_image_prompt(data, beat_dict)
                    img_bytes = gen_image_b64(final_image_prompt, size=image_size)
                    img_path = base_dir / "images" / f"page_{beat.page:02d}.png"
                    save_bytes(img_path, img_bytes)

                page_record = {
                    "page": beat.page,
                    "summary": beat.summary,
                    "text": page_text,
                    "image_prompt_final": final_image_prompt,
                }
                generated_pages.append(page_record)
                write_text(base_dir / "pages" / f"page_{beat.page:02d}.json", json.dumps(page_record, indent=2))

        status_text.text("✅ Story complete!")
        progress_bar.empty()

        # 调试信息：验证生成的文件
        print(f"🔍 故事生成完成，验证文件状态:")
        print(f"   - 基础目录: {base_dir}")
        print(f"   - 生成页面数: {len(generated_pages)}")
        
        # 检查图片文件
        images_dir = base_dir / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            print(f"   - 图片目录存在，找到 {len(image_files)} 个PNG文件")
            for img_file in image_files:
                print(f"     - {img_file.name}: {img_file.stat().st_size} bytes")
        else:
            print(f"   - ❌ 图片目录不存在: {images_dir}")
        
        # 检查页面文件
        pages_dir = base_dir / "pages"
        if pages_dir.exists():
            page_files = list(pages_dir.glob("*.json"))
            print(f"   - 页面目录存在，找到 {len(page_files)} 个JSON文件")
        else:
            print(f"   - ❌ 页面目录不存在: {pages_dir}")

        # Persist run outputs to session for later reruns (export buttons)
        ss.story_ready = True
        ss.generated_pages = generated_pages
        ss.base_dir = base_dir
        ss.folder_slug = folder_slug
        ss.image_size = image_size

        # Immediately show preview after generation
        st.subheader("📚 Story Preview")

        # 根据图片尺寸计算翻书尺寸
        try:
            w, h = map(int, str(image_size).lower().split("x"))
        except Exception:
            w, h = 1024, 1024
        book_w = int(max(w * 1.9, 900))
        book_h = int(max(h * 0.9, 600))

        title = (folder_slug or "").replace("-", " ").title()
        author = None

        # 直接使用翻书模式，不再提供选择
        render_flipbook_spread(generated_pages, base_dir, book_w=book_w, book_h=book_h,
                               title=title, author=author)

        # Show export options
        st.subheader("📦 Export Your Story")
        
        # Initialize export state in session
        if 'export_pdf_data' not in st.session_state:
            st.session_state.export_pdf_data = None
        if 'export_zip_data' not in st.session_state:
            st.session_state.export_zip_data = None
        if 'export_pdf_filename' not in st.session_state:
            st.session_state.export_pdf_filename = None
        if 'export_zip_filename' not in st.session_state:
            st.session_state.export_zip_filename = None
        if 'export_pdf_ready' not in st.session_state:
            st.session_state.export_pdf_ready = False
        if 'export_zip_ready' not in st.session_state:
            st.session_state.export_zip_ready = False
        
        colA, colB = st.columns(2)
        with colA:
            # PDF Export
            if st.button("📄 Export PDF", key="export_pdf_btn"):
                try:
                    with st.spinner("Creating PDF..."):
                        pdf_path = export_pdf(base_dir, generated_pages, image_size)
                        if pdf_path.exists():
                            # Read file data and store in session
                            with open(pdf_path, "rb") as f:
                                st.session_state.export_pdf_data = f.read()
                            st.session_state.export_pdf_filename = f"{folder_slug}_storybook.pdf"
                            st.session_state.export_pdf_ready = True
                            st.success(f"✅ PDF created successfully!")
                            st.rerun()  # Rerun to show download button
                        else:
                            st.error("❌ PDF file was not created")
                except Exception as e:
                    st.error(f"❌ Error creating PDF: {str(e)}")
                    st.info("💡 Check if the output directory exists and has write permissions")
            
            # Show download button if PDF is ready
            if st.session_state.export_pdf_ready and st.session_state.export_pdf_data is not None:
                st.download_button(
                    "📥 Download PDF", 
                    data=st.session_state.export_pdf_data,
                    file_name=st.session_state.export_pdf_filename, 
                    mime="application/pdf",
                    key="download_pdf_btn"
                )
                # Clear button
                if st.button("Clear PDF", key="clear_pdf_btn"):
                    st.session_state.export_pdf_data = None
                    st.session_state.export_pdf_filename = None
                    st.session_state.export_pdf_ready = False
                    st.rerun()
                    
        with colB:
            # ZIP Export
            if st.button("📁 Export ZIP", key="export_zip_btn"):
                try:
                    with st.spinner("Creating ZIP..."):
                        zip_path = export_zip(base_dir)
                        if zip_path.exists():
                            # Read file data and store in session
                            with open(zip_path, "rb") as f:
                                st.session_state.export_zip_data = f.read()
                            st.session_state.export_zip_filename = f"{folder_slug}_story_export.zip"
                            st.session_state.export_zip_ready = True
                            st.success(f"✅ ZIP created successfully!")
                            st.rerun()  # Rerun to show download button
                        else:
                            st.error("❌ ZIP file was not created")
                except Exception as e:
                    st.error(f"❌ Error creating ZIP: {str(e)}")
                    st.info("💡 Check if the output directory exists and has write permissions")
            
            # Show download button if ZIP is ready
            if st.session_state.export_zip_ready and st.session_state.export_zip_data is not None:
                st.download_button(
                    "📥 Download ZIP", 
                    data=st.session_state.export_zip_data,
                    file_name=st.session_state.export_zip_filename, 
                    mime="application/zip",
                    key="download_zip_btn"
                )
                # Clear button
                if st.button("Clear ZIP", key="clear_zip_btn"):
                    st.session_state.export_zip_data = None
                    st.session_state.export_zip_filename = None
                    st.session_state.export_zip_ready = False
                    st.rerun()

    except Exception as e:
        st.error(f"❌ Error generating story: {str(e)}")
        st.info("💡 Tip: Check your API key and internet connection. If the error persists, try simplifying your story idea.")
