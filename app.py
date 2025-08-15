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
from fpdf import FPDF

from utils import (
    llm_text,
    gen_image_b64,
    write_text,
    save_bytes,
    StoryBible,
    IMAGE_SIZE,
)

# ---------- Helpers ----------
def slugify(text: str) -> str:
    import re as _re
    s = _re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return _re.sub(r"-+", "-", s).strip("-") or "story"

def ensure_dirs(base: Path) -> None:
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "pages").mkdir(parents=True, exist_ok=True)

def export_pdf(base_dir: Path, pages: List[Dict[str, Any]], image_size: str = IMAGE_SIZE) -> Path:
    w, h = map(int, image_size.split("x"))
    pdf = FPDF(orientation="P", unit="pt", format=(w, h + 200))
    pdf.set_auto_page_break(auto=False)
    for i, page in enumerate(pages, start=1):
        pdf.add_page()
        img_path = base_dir / "images" / f"page_{i:02d}.png"
        if img_path.exists():
            pdf.image(str(img_path), x=0, y=0, w=w, h=h)
        pdf.set_xy(36, h + 24)
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(w - 72, 16, txt=page.get("text", "")[:1200])
    out_path = base_dir / "book.pdf"
    pdf.output(str(out_path))
    return out_path

def export_zip(base_dir: Path) -> Path:
    import zipfile
    out_path = base_dir / "story_export.zip"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in base_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(base_dir))
    return out_path

# ---------- Prompt templates (STUDENTS: Improve these!) ----------
def make_bible_prompt(idea: str, pages: int, schema: dict) -> str:
    """
    STUDENT TASK: Improve this prompt to create better story bibles!
    Consider adding:
    - Genre-specific guidance
    - Character development rules
    - Plot structure templates
    - Age-appropriate content guidelines
    """
    return f"""
You are the Architect. Turn IDEA + TARGET_PAGES into a Story Bible JSON.

Rules:
- Output ONLY JSON (no backticks, no prose).
- The JSON must match the schema (fields and types).
- Include plot_beats with exactly TARGET_PAGES items, pages numbered 1..N.
- Each beat must have summary and image_prompt.
- Keep character visual_anchors consistent across all pages.
- Create a cohesive story arc with beginning, middle, and end.
- Ensure each page advances the plot meaningfully.

IDEA: {idea}
TARGET_PAGES: {pages}

SCHEMA:
{json.dumps(schema)}
""".strip()

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
            prompt = make_bible_prompt(idea, int(pages), schema)
            raw = llm_text(prompt, response_format={"type": "json_object"}, temperature=0.4)
            data = json.loads(raw)
            bible = StoryBible(**data)  # validates structure

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
                # Author draft
                page_text = llm_text(make_page_text_prompt(data, beat_dict), temperature=0.7)

                # Designer prompt + image
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

        # Preview
        st.subheader("📚 Story Preview")
        for page in generated_pages:
            with st.container(border=True):
                st.markdown(f"**Page {page['page']}** — {page['summary']}")
                st.image(str(base_dir / "images" / f"page_{page['page']:02d}.png"), caption="Illustration")
                st.markdown("**Text**")
                st.write(page["text"])
                with st.expander("🎨 Image prompt (final)"):
                    st.code(page["image_prompt_final"])

        # Exports
        st.subheader("📦 Export Your Story")
        colA, colB = st.columns(2)
        with colA:
            if st.button("📄 Export PDF"):
                pdf_path = export_pdf(base_dir, generated_pages, image_size)
                st.success(f"PDF created: {pdf_path}")
                st.download_button("Download PDF", data=open(pdf_path, "rb").read(),
                                   file_name=pdf_path.name, mime="application/pdf")
        with colB:
            if st.button("📁 Export ZIP"):
                zip_path = export_zip(base_dir)
                st.success(f"ZIP created: {zip_path}")
                st.download_button("Download ZIP", data=open(zip_path, "rb").read(),
                                   file_name=zip_path.name, mime="application/zip")
                                   
    except Exception as e:
        st.error(f"❌ Error generating story: {str(e)}")
        st.info("💡 Tip: Check your API key and internet connection. If the error persists, try simplifying your story idea.")
