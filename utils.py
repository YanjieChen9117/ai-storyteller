import os
import base64
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please create a .env file with your API key. See .env.example for an example.")

# Initialize Gemini client
_client = genai.Client(api_key=api_key)

# Initialize Imagen client (if not exists)
_genai_client = genai.Client()  # Read GEMINI_API_KEY

# --- Story Bible types (students may expand) ---
class Character(BaseModel):
    """A character in the story with consistent visual and personality traits."""
    name: str = Field(..., description="Character's name")
    role: str = Field(..., description="Character's role in the story (protagonist, mentor, etc.)")
    personality: str = Field(..., description="Key personality traits and behaviors")
    visual_anchors: List[str] = Field(default_factory=list, description="Consistent visual elements (clothing, features, etc.)")

class PlotBeat(BaseModel):
    """A single page/scene in the story with text and image requirements."""
    page: int = Field(..., description="Page number (1-based)")
    summary: str = Field(..., description="Brief summary of what happens on this page")
    image_prompt: str = Field(..., description="Description of the scene to illustrate")

class StoryBible(BaseModel):
    """Complete story specification with characters, style, and plot structure."""
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata like title, target audience, etc.")
    themes: List[str] = Field(default_factory=list, description="Main themes of the story")
    tone: str = Field(default="Warm, imaginative, hopeful.", description="Overall emotional tone")
    narrator_voice: str = Field(default="Third-person, gentle, playful.", description="Narrator's voice and style")
    world: str = Field(default="", description="Setting and world-building details")
    characters: List[Character] = Field(default_factory=list, description="All characters in the story")
    art_style: Dict[str, Any] = Field(default_factory=dict, description="Visual style specifications")
    continuity_rules: List[str] = Field(default_factory=list, description="Rules to maintain consistency")
    plot_beats: List[PlotBeat] = Field(default_factory=list, description="All pages/scenes in order")

# --- API wrappers (students implement) ---
def llm_text(prompt: str,
             temperature: float = 0.2,
             model: Optional[str] = None, 
             max_tokens: int = 256) -> str:
    """
    Single-shot text call with automatic retries using Gemini. Students provide the full prompt.
    
    Args:
        prompt: The complete prompt (system/user style OK)
        temperature: Creativity level (0.0 = deterministic, 1.0 = very creative)
        model: Gemini model to use (defaults to MODEL_TEXT env var)
        max_tokens: Maximum output tokens
    
    Returns:
        The model's text response
        
    Raises:
        Exception: If API call fails after retries
    """
    # Simple retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            m = model or "gemini-2.5-flash-lite"
            r = _client.models.generate_content(
                model=m,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            return r.text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise Exception(f"Gemini API call failed: {str(e)}")

def llm_json(prompt: str,
             model: Optional[str] = None,
             temperature: float = 0.2,
             max_tokens: int = 2048) -> dict:
    """
    JSON-only text call with automatic retries using Gemini. Forces strict JSON output.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            m = model or "gemini-2.5-flash-lite"
            enhanced_prompt = f"""You must respond with ONLY valid JSON. No explanations, no markdown formatting, just pure JSON.

{prompt}

Remember: Return ONLY the JSON object, nothing else."""
            resp = _client.models.generate_content(
                model=m,
                contents=enhanced_prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            text = resp.text or ""
            if not text.strip():
                raise ValueError("Empty response from Gemini")
            # Strip ```json ... ``` blocks
            mobj = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
            if mobj:
                text = mobj.group(1)
            # Direct parse
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                cleaned_text = text.strip()
                start_idx = cleaned_text.find('{')
                end_idx = cleaned_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    cleaned_text = cleaned_text[start_idx:end_idx + 1]
                try:
                    return json.loads(cleaned_text)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format. Response: {text[:200]}...")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                import time
                time.sleep(2 ** attempt)
                continue
            else:
                if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                    raise Exception("Gemini API authentication failed. Please check your GEMINI_API_KEY.")
                elif "quota" in str(e).lower() or "limit" in str(e).lower():
                    raise Exception("Gemini API quota exceeded. Please try again later.")
                elif "network" in str(e).lower() or "connection" in str(e).lower():
                    raise Exception("Network connection failed. Please check your internet connection.")
                else:
                    raise Exception(f"Gemini API error: {str(e)}")

def gen_image_b64(prompt: str,
                  model: str = "gemini-2.0-flash-exp",
                  size: str = "1024x1024") -> bytes:
    try:
        try:
            w, h = map(int, size.lower().split("x"))
        except Exception:
            w, h = 1024, 1024
        aspect = "1:1" if w == h else ("16:9" if w > h else "9:16")
        image_size = "1K" if max(w, h) <= 1024 else "2K"

        model_id = os.getenv("MODEL_IMAGE", "imagen-4.0-generate-001")
        resp = _genai_client.models.generate_images(
            model=model_id,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                image_size=image_size,   # Correct field name
                aspect_ratio=aspect,
            ),
        )
        png = resp.generated_images[0].image.image_bytes  # PNG bytes

        # Normalize to requested size to avoid downstream validation/display issues
        try:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(png)) as im:
                if im.size != (w, h):
                    im = im.convert("RGBA") if im.mode != "RGBA" else im
                    buf = io.BytesIO()
                    im.resize((w, h), Image.LANCZOS).save(buf, format="PNG")
                    png = buf.getvalue()
        except Exception:
            pass

        return png

    except Exception as e:
        print(f"[Imagen4 fallback] {type(e).__name__}: {e}")
        # Placeholder fallback
        try:
            return placeholder_image(prompt, size)
        except Exception:
            # Secondary fallback: solid-color PNG
            from io import BytesIO
            from PIL import Image
            buf = BytesIO()
            try:
                w, h = map(int, size.lower().split("x"))
            except Exception:
                w, h = 1024, 1024
            Image.new("RGB", (w, h), (240, 248, 255)).save(buf, format="PNG")
            return buf.getvalue()

def placeholder_image(prompt: str, size: str = "1024x1024") -> bytes:
    """
    Generate a placeholder image when Imagen generation fails.
    This provides a visual fallback with informative text.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        width, height = map(int, size.split("x"))
        img = Image.new('RGB', (width, height), color='#87CEEB')  # Sky blue
        draw = ImageDraw.Draw(img)

        # Sun
        draw.ellipse([width-100, 20, width-20, 100], fill='#FFD700')
        # Clouds
        for i in range(3):
            x = 50 + i * 150
            y = 50 + i * 20
            draw.ellipse([x, y, x+80, y+40], fill='white')
            draw.ellipse([x+20, y+20, x+100, y+60], fill='white')

        # Text
        try:
            font_size = min(width // 20, 24)
            font = ImageFont.load_default()
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                except:
                    pass
        except:
            font = ImageFont.load_default()

        placeholder_text = "AI Image Generation"
        text_bbox = draw.textbbox((0, 0), placeholder_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = height // 2 - 30
        draw.text((text_x, text_y), placeholder_text, fill='white', font=font, stroke_width=2, stroke_fill='black')

        subtitle = "Placeholder - Imagen generation failed"
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (width - subtitle_width) // 2
        subtitle_y = height // 2 + 10
        draw.text((subtitle_x, subtitle_y), subtitle, fill='white', font=font, stroke_width=1, stroke_fill='black')

        prompt_preview = prompt[:30] + "..." if len(prompt) > 30 else prompt
        draw.text((20, height-60), f"Prompt: {prompt_preview}", fill='black', font=font)

        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    except Exception as e:
        raise Exception(f"Placeholder image generation failed: {str(e)}")

# --- Configuration ---
MODEL_TEXT = "gemini-2.5-flash-lite"
IMAGE_SIZE = "700x700"

# --- Utility functions ---
def save_bytes(path: Path, data: bytes) -> None:
    """Save binary data to a file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True
    )
    path.write_bytes(data)

def write_text(path: Path, text: str) -> None:
    """Save text to a file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# --- Architect functions for Story Bible generation ---
def validate_bible(data: dict) -> tuple[bool, Optional[str]]:
    """Validate structure using StoryBible(**data)."""
    try:
        StoryBible(**data)
        return True, None
    except Exception as e:
        return False, str(e)

def make_bible_repair_prompt(schema: dict, idea: str, pages: int, prev: dict, error: str) -> str:
    """Generate repair prompt for invalid Story Bible output."""
    return f"""The previous Story Bible generation failed with this error: {error}

Previous attempt (fix this JSON):
{json.dumps(prev, indent=2, ensure_ascii=False)}

IMPORTANT: You must create a STORY, not a schema definition. The response should contain actual story content.

REQUIREMENTS:
1. Output ONLY valid JSON - no markdown, no explanations, no extra text
2. Create a complete story with {pages} pages
3. plot_beats array MUST contain exactly {pages} items with actual story content
4. Each plot_beat must have: {{ "page": n, "summary": "...(<=20 words)", "image_prompt": "...(<=30 words)" }}
5. Include art_style with: style_tags (array), palette (array), composition_rules (string)
6. Create characters with names, roles, and visual descriptions
7. Set a story title in meta.title

STORY STRUCTURE:
- meta: {{ "title": "Story Title", "target_audience": "children" }}
- characters: List of characters with names, roles, personalities, visual_anchors
- art_style: {{ "style_tags": ["watercolor", "whimsical"], "palette": ["warm", "bright"], "composition_rules": "storybook framing" }}
- plot_beats: Array of {pages} story pages with summaries and image prompts

Story idea: {idea}
Target pages: {pages}

Fix the JSON and return ONLY the corrected version with actual story content:"""

def make_bible_prompt_local(idea: str, pages: int, schema: dict) -> str:
    """Local equivalent of app.make_bible_prompt to avoid circular import."""
    return f"""You are The Architect - a master storyteller who creates structured story plans.

Create a Story Bible for: "{idea}"
Target length: {pages} pages

IMPORTANT: You must create a STORY, not a schema definition. The response should contain actual story content.

REQUIREMENTS:
1. Output ONLY valid JSON - no markdown, no explanations, no extra text
2. Create a complete story with {pages} pages
3. plot_beats array MUST contain exactly {pages} items with actual story content
4. Each plot_beat must have: {{ "page": n, "summary": "...(<=20 words)", "image_prompt": "...(<=30 words)" }}
5. Include art_style with: style_tags (array), palette (array), composition_rules (string)
6. Create characters with names, roles, and visual descriptions
7. Set a story title in meta.title

STORY STRUCTURE:
- meta: {{ "title": "Story Title", "target_audience": "children" }}
- characters: List of characters with names, roles, personalities, visual_anchors
- art_style: {{ "style_tags": ["watercolor", "whimsical"], "palette": ["warm", "bright"], "composition_rules": "storybook framing" }}
- plot_beats: Array of {pages} story pages with summaries and image prompts

Focus on:
- Clear character development
- Logical plot progression
- Visual consistency
- Age-appropriate themes

Return ONLY the JSON object with the actual story content:"""

def ensure_bible(idea: str, pages: int, schema: dict, max_attempts: int = 3, temperature: float = 0.3):
    """
    生成→校验；若失败就以错误报告+上次 JSON 让模型修复，最多 max_attempts 次。
    成功返回 (StoryBible实例, 原始dict)。
    """
    prev_data: Dict[str, Any] = {}
    error_msg: str = ""
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                prompt = make_bible_prompt_local(idea, pages, schema)
                data = llm_json(prompt, temperature=temperature)
            else:
                prompt = make_bible_repair_prompt(schema, idea, pages, prev_data, error_msg)
                data = llm_json(prompt, temperature=temperature)

            is_valid, val_err = validate_bible(data)
            if is_valid:
                bible = StoryBible(**data)
                return bible, data

            prev_data = data
            error_msg = val_err or "Validation failed without details."
        except Exception as e:
            prev_data = (locals().get("data") or prev_data) or {}
            error_msg = str(e)
            if attempt == max_attempts - 1:
                raise Exception(f"Story Bible generation failed after {max_attempts} attempts. Last error: {error_msg}")
            continue
    raise Exception(f"Story Bible generation failed after {max_attempts} attempts.")

# --- Designer functions for image generation and validation ---
def build_style_pack(bible: dict) -> str:
    """Extract style info from bible['art_style'] and return concise style string."""
    art_style = bible.get('art_style', {})
    style_parts = []
    style_tags = art_style.get('style_tags', [])
    if style_tags:
        style_parts.append(f"style: {', '.join(style_tags[:3])}")
    palette = art_style.get('palette', [])
    if palette:
        style_parts.append(f"palette: {', '.join(palette[:4])}")
    composition = art_style.get('composition_rules', '')
    if composition:
        comp_text = composition[:50] + "..." if len(composition) > 50 else composition
        style_parts.append(f"composition: {comp_text}")
    if not style_parts:
        return "storybook illustration, consistent visual style"
    return "; ".join(style_parts)

def build_image_prompt(bible: dict, beat: dict) -> str:
    """
    TODO[Designer]: Combine style_pack + character anchors + scene.
    Goal: ensure cross-page consistency and include palette/style in prompt.
    Requirements:
    - Extract style_tags/palette/composition_rules from bible['art_style']
    - Aggregate characters.visual_anchors (dedupe, at most 3)
    - Merge beat['image_prompt'] scene, limit length <= 150
    Baseline: return scene-only without style/anchors.
    """
    scene = beat.get('image_prompt', '')
    if len(scene) > 150:
        scene = scene[:147] + "..."
    return f"storybook illustration. Scene: {scene}"

def analyze_image_bytes(img_bytes: bytes) -> Dict[str, Any]:
    """Use PIL to read image; return width, height, aspect, unique_colors(approx), entropy(rough), format."""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size
        aspect = width / height if height > 0 else 0
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pixels = list(img.getdata())
        sample_size = min(1000, len(pixels))
        sample_pixels = pixels[::len(pixels)//sample_size] if len(pixels) > sample_size else pixels
        unique_colors = len(set(sample_pixels))
        if sample_pixels:
            r_values = [p[0] for p in sample_pixels]
            g_values = [p[1] for p in sample_pixels]
            b_values = [p[2] for p in sample_pixels]
            def variance(values):
                if not values:
                    return 0
                mean = sum(values) / len(values)
                return sum((x - mean) ** 2 for x in values) / len(values)
            entropy = (variance(r_values) + variance(g_values) + variance(b_values)) / 3
        else:
            entropy = 0
        return {
            'width': width,
            'height': height,
            'aspect': round(aspect, 3),
            'unique_colors': unique_colors,
            'entropy': round(entropy, 1),
            'format': img.format or 'unknown'
        }
    except Exception as e:
        return {
            'error': str(e),
            'width': 0,
            'height': 0,
            'aspect': 0,
            'unique_colors': 0,
            'entropy': 0,
            'format': 'error'
        }

# --- Advanced metrics scaffolding (students extend) ---
def compute_readability_metrics(text: str) -> Dict[str, Any]:
    """TODO[Author-Advanced]: Compute readability metrics (e.g., FK grade, lexical diversity, syllables). Placeholder returns coarse counts."""
    sentences = _split_sentences(text)
    word_count = _word_count(text)
    avg_words_per_sentence = round(word_count / max(len(sentences), 1), 1)
    # TODO: 计算 grade_level / lexical_diversity / syllables_per_word
    return {
        "avg_words_per_sentence": avg_words_per_sentence,
        "sentence_count": len(sentences),
        "word_count": word_count,
        "grade_level": None,  # TODO: 实现后填充
        "lexical_diversity": None,  # TODO: 实现后填充
        "avg_syllables_per_word": None,  # TODO: 实现后填充
    }

def detect_repetition(text: str) -> Dict[str, Any]:
    """TODO[Author-Advanced]: Detect repeated phrases/high-frequency n-grams; return score and examples. Placeholder uses simple n-gram counts."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    n = 2
    ngrams = [" ".join(tokens[i:i+n]) for i in range(max(len(tokens)-n+1, 0))]
    freq: Dict[str, int] = {}
    for g in ngrams:
        freq[g] = freq.get(g, 0) + 1
    repeated = sorted([k for k, v in freq.items() if v >= 3], key=lambda k: -freq[k])[:5]
    score = sum(freq[k] for k in repeated)
    return {"repeated_phrases": repeated, "repetition_score": score}

def extract_palette_from_image(img_bytes: bytes, max_colors: int = 5) -> List[Tuple[int, int, int]]:
    """TODO[Designer-Advanced]: Extract dominant palette (RGB). Suggest PIL.quantize or KMeans. Current impl uses simplified quantize."""
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(img_bytes)) as im:
            im = im.convert("RGB")
            q = im.quantize(colors=max_colors, method=2)
            palette = q.getpalette()[: max_colors * 3]
            colors = [(palette[i], palette[i+1], palette[i+2]) for i in range(0, len(palette), 3)]
            return colors[:max_colors]
    except Exception:
        return []

def _hex_to_rgb(hex_or_name: str) -> Optional[Tuple[int, int, int]]:
    try:
        s = hex_or_name.strip().lstrip('#').lower()
        if len(s) in (3, 6) and all(c in '0123456789abcdef' for c in s):
            if len(s) == 3:
                s = ''.join(ch*2 for ch in s)
            return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except Exception:
        pass
    return None

def compare_palette(target_palette: List[str], image_palette: List[Tuple[int, int, int]], tolerance: float = 0.28) -> float:
    """TODO[Designer-Advanced]: Compare target palette (name/hex) vs image dominant colors; return [0,1] score. Current: min distance average."""
    target_rgbs: List[Tuple[int, int, int]] = []
    for t in target_palette or []:
        rgb = _hex_to_rgb(str(t))
        if rgb:
            target_rgbs.append(rgb)
    if not target_rgbs or not image_palette:
        return 0.5  # 中性分数
    import math
    def dist(a, b):
        return math.sqrt(sum((ax-bx)**2 for ax, bx in zip(a, b))) / 441.67  # 255*sqrt(3) 近似归一化
    dists = []
    for tr in target_rgbs:
        best = min(dist(tr, ir) for ir in image_palette)
        dists.append(best)
    avg = sum(dists) / len(dists)
    return max(0.0, 1.0 - avg / max(tolerance, 1e-6))

def validate_image(img_bytes: bytes, expected_size: str, anchors: List[str], strict: bool = True,
                   check_palette: bool = False, target_palette: Optional[List[str]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    - Size must match expected_size
    - Basic complexity check: fail in strict mode if unique_colors/entropy too low
    - Anchors currently relaxed
    """
    try:
        expected_w, expected_h = map(int, expected_size.split("x"))
    except:
        return False, f"Invalid expected_size format: {expected_size}", {}
    metrics = analyze_image_bytes(img_bytes)
    if 'error' in metrics:
        return False, f"Image analysis failed: {metrics['error']}", metrics
    if metrics['width'] != expected_w or metrics['height'] != expected_h:
        return False, f"Size mismatch: got {metrics['width']}x{metrics['height']}, expected {expected_size}", metrics
    if strict:
        if metrics['unique_colors'] < 10:
            return False, f"Low complexity detected: only {metrics['unique_colors']} colors (possible placeholder)", metrics
        # Relax threshold to avoid frequent false negatives on placeholders
        if metrics['entropy'] < 20:
            return False, f"Low entropy detected: {metrics['entropy']} (possible placeholder)", metrics
    # Optional: palette consistency check
    if check_palette:
        # TODO[Designer-Advanced]: Improve palette extraction/matching (KMeans, hue distance, weighted similarity)
        image_palette = extract_palette_from_image(img_bytes, max_colors=5)
        score = compare_palette(target_palette or [], image_palette)
        metrics['palette_score'] = round(score, 3)
        if strict and score < 0.5:
            return False, f"Palette mismatch (score={score:.2f})", metrics
    # Anchors check (currently relaxed)
    return True, "Image validation passed", metrics

def ensure_image(bible: dict, beat: dict, size: str, max_attempts: int = 2, strict: bool = True,
                 enforce_palette: bool = False) -> Tuple[bytes, str, Dict[str, Any]]:
    """Generate image -> validate; retry if needed. Return (img_bytes, final_prompt, metrics)."""
    anchors = [a for c in bible.get('characters', []) for a in c.get('visual_anchors', [])]
    reason = "unknown error"
    for attempt in range(max_attempts):
        try:
            prompt = build_image_prompt(bible, beat)
            img_bytes = gen_image_b64(prompt, size=size)
            target_palette = (bible.get('art_style') or {}).get('palette') or []
            ok, reason, metrics = validate_image(
                img_bytes, size, anchors, strict,
                check_palette=enforce_palette, target_palette=target_palette
            )
            if ok:
                return img_bytes, prompt, metrics
            if strict and attempt < max_attempts - 1:
                print(f"Image validation failed (attempt {attempt + 1}): {reason}. Retrying...")
                continue
            elif not strict:
                return img_bytes, prompt, metrics
        except Exception as e:
            reason = str(e)
            if attempt < max_attempts - 1:
                print(f"Image generation failed (attempt {attempt + 1}): {reason}. Retrying...")
                continue
            else:
                raise RuntimeError(f"Image generation failed after {max_attempts} attempts: {reason}")
    raise RuntimeError(f"Image generation failed after {max_attempts} attempts: {reason}")

# --- Author functions for page text generation and validation ---
def build_page_text_prompt(bible: dict, beat: dict) -> str:
    """Author final prompt (return only final text; no title/markdown)."""
    tone = bible.get('tone', 'Warm, imaginative, hopeful.')
    narrator_voice = bible.get('narrator_voice', 'Third-person, gentle, playful.')
    character_names = [char.get('name', '') for char in bible.get('characters', [])]
    character_text = f"Characters: {', '.join(character_names)}" if character_names else ""
    summary = beat.get('summary', '')
    return f"""You are The Author - a master storyteller who writes engaging page text.

Write the final page text for this story beat. Return ONLY the story text - no title, no markdown, no explanations.

REQUIREMENTS:
- Tone: {tone}
- Voice: {narrator_voice}
- Length: 2-4 sentences, maximum 90 words
- Content: Must include elements from this summary: "{summary}"
- Characters: Use these names consistently: {character_text}
- Style: Engaging, age-appropriate, avoid complex vocabulary
- Ending: Do NOT end with a question mark (avoid asking readers directly)

Write the page text now:"""

def _split_sentences(text: str) -> List[str]:
    """Roughly split sentences by regex (. ! ?), trim whitespace."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def _word_count(text: str) -> int:
    """Approximate word count; for CJK treat characters approximately."""
    words = re.findall(r'\b\w+\b', text.lower())
    return len(words)

# Light forbidden words example, can be expanded as needed
FORBIDDEN = {"bloody", "damn", "kill", "sex", "hate", "stupid", "ugly"}

def validate_page_text(text: str, bible: dict, beat: dict, strict: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (passed, reason, metrics)."""
    cleaned_text = text.strip()
    sentences = _split_sentences(cleaned_text)
    sentence_count = len(sentences)
    word_count = _word_count(cleaned_text)
    ends_with_question = cleaned_text.rstrip().endswith('?')
    character_names = [char.get('name', '').lower() for char in bible.get('characters', [])]
    has_main_character = any(name in cleaned_text.lower() for name in character_names if name)
    text_lower = cleaned_text.lower()
    forbidden_found = [word for word in FORBIDDEN if word in text_lower]
    avg_words_per_sentence = round(word_count / sentence_count, 1) if sentence_count > 0 else 0

    if strict:
        if sentence_count < 2:
            return False, f"Too few sentences: {sentence_count} (need 2-4)", {
                'sentence_count': sentence_count,
                'word_count': word_count,
                'avg_words_per_sentence': avg_words_per_sentence,
                'has_main_character': has_main_character,
                'forbidden_found': forbidden_found,
                'ends_with_question': ends_with_question
            }
        if sentence_count > 4:
            return False, f"Too many sentences: {sentence_count} (need 2-4)", {
                'sentence_count': sentence_count,
                'word_count': word_count,
                'avg_words_per_sentence': avg_words_per_sentence,
                'has_main_character': has_main_character,
                'forbidden_found': forbidden_found,
                'ends_with_question': ends_with_question
            }
        if word_count > 90:
            return False, f"Too many words: {word_count} (max 90)", {
                'sentence_count': sentence_count,
                'word_count': word_count,
                'avg_words_per_sentence': avg_words_per_sentence,
                'has_main_character': has_main_character,
                'forbidden_found': forbidden_found,
                'ends_with_question': ends_with_question
            }
        if ends_with_question:
            return False, "Text ends with a question mark (should not ask readers directly)", {
                'sentence_count': sentence_count,
                'word_count': word_count,
                'avg_words_per_sentence': avg_words_per_sentence,
                'has_main_character': has_main_character,
                'forbidden_found': forbidden_found,
                'ends_with_question': ends_with_question
            }
        if not has_main_character:
            return False, "No main character names found in text", {
                'sentence_count': sentence_count,
                'word_count': word_count,
                'avg_words_per_sentence': avg_words_per_sentence,
                'has_main_character': has_main_character,
                'forbidden_found': forbidden_found,
                'ends_with_question': ends_with_question
            }
        if forbidden_found:
            return False, f"Forbidden words found: {', '.join(forbidden_found)}", {
                'sentence_count': sentence_count,
                'word_count': word_count,
                'avg_words_per_sentence': avg_words_per_sentence,
                'has_main_character': has_main_character,
                'forbidden_found': forbidden_found,
                'ends_with_question': ends_with_question
            }

    return True, "Text validation passed", {
        'sentence_count': sentence_count,
        'word_count': word_count,
        'avg_words_per_sentence': avg_words_per_sentence,
        'has_main_character': has_main_character,
        'forbidden_found': forbidden_found,
        'ends_with_question': ends_with_question
    }

def make_page_text_repair_prompt(bible: dict, beat: dict, previous_text: str, reason: str) -> str:
    """Turn validation failures into repair instructions; include previous_text."""
    return f"""The previous page text failed validation: {reason}

Previous text:
"{previous_text}"

Please rewrite this page text, keeping the same story elements but fixing the validation issues.
Return ONLY the corrected story text - no explanations, no markdown, no title.

REQUIREMENTS:
- Tone: {bible.get('tone', 'Warm, imaginative, hopeful.')}
- Voice: {bible.get('narrator_voice', 'Third-person, gentle, playful.')}
- Length: 2-4 sentences, maximum 90 words
- Content: Must include elements from: "{beat.get('summary', '')}"
- Characters: Use these names: {[char.get('name', '') for char in bible.get('characters', [])]}
- Style: Engaging, age-appropriate, avoid complex vocabulary
- Ending: Do NOT end with a question mark

Write the corrected page text:"""

def ensure_page_text(bible: dict, beat: dict, max_attempts: int = 2, strict: bool = True, temperature: float = 0.7) -> Tuple[str, Dict[str, Any]]:
    """Loop: prompt -> llm_text -> clean -> validate; on failure, attempt repair."""
    prev_text = ""
    error_reason = ""
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                prompt = build_page_text_prompt(bible, beat)
                text = llm_text(prompt, temperature=temperature)
            else:
                prompt = make_page_text_repair_prompt(bible, beat, prev_text, error_reason)
                text = llm_text(prompt, temperature=temperature)

            cleaned_text = text.strip()
            cleaned_text = re.sub(r'```.*?```', '', cleaned_text, flags=re.DOTALL).strip()

            is_valid, error_reason, metrics = validate_page_text(cleaned_text, bible, beat, strict)
            if is_valid:
                # TODO[Author]: 你可以扩展 validate_page_text 的规则（如阅读等级/重复用词检测），
                # 并在此返回更丰富的 metrics（保持返回结构不变）。
                return cleaned_text, metrics

            prev_text = cleaned_text
        except Exception as e:
            error_reason = str(e)
            prev_text = locals().get('text', '') or prev_text
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Page text generation failed after {max_attempts} attempts: {error_reason}")
            continue
    raise RuntimeError(f"Page text generation failed after {max_attempts} attempts: {error_reason}")
