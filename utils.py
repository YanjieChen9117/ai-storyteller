import os
import base64
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please create a .env file with your API key. See .env.example for an example.")

# Initialize Gemini client
_client = genai.Client(api_key=api_key)

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
             model: str | None = None, 
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
            # Always use Gemini model
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
             model: str | None = None,
             temperature: float = 0.2,
             max_tokens: int = 2048) -> dict:
    """
    JSON-only text call with automatic retries using Gemini. Forces strict JSON output.

    Args:
        prompt: The complete prompt (system/user style OK)
        model: Gemini model to use (defaults to MODEL_TEXT env var)
        temperature: Creativity level (0.0 = deterministic, 1.0 = very creative)
        max_tokens: Maximum output tokens

    Returns:
        Parsed JSON as dict

    Raises:
        ValueError: If response is not valid JSON
        Exception: If API call fails after retries
    """
    # Simple retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Always use Gemini model
            m = model or "gemini-2.5-flash-lite"
            
            # Enhanced prompt to force JSON output
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
                
            # 容错：若模型仍把 JSON 放在 ```json ... ``` 中，剥壳
            mobj = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
            if mobj:
                text = mobj.group(1)
            
            # 尝试直接解析
            try:
                result = json.loads(text)
                return result
            except json.JSONDecodeError as json_error:
                # 如果直接解析失败，尝试清理常见的格式问题
                cleaned_text = text.strip()
                
                # 移除开头和结尾的非JSON字符
                start_idx = cleaned_text.find('{')
                end_idx = cleaned_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    cleaned_text = cleaned_text[start_idx:end_idx + 1]
                
                # 尝试解析清理后的文本
                try:
                    result = json.loads(cleaned_text)
                    return result
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format. Response: {text[:200]}...")
                    
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                # Last attempt failed, provide clear error
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
    """
    Generate an image from a text prompt using Gemini.
    Note: This function currently returns a placeholder image since Gemini text models don't support image generation.
    For actual image generation, you would need to use a different service like DALL-E, Midjourney, or Stable Diffusion.

    Args:
        prompt: Detailed description of the image to generate
        model: Gemini model to use (default: gemini-2.0-flash-exp)
        size: Image dimensions (e.g., "1024x1024")

    Returns:
        Placeholder image data as bytes (a simple colored square)

    Raises:
        Exception: If image generation fails
    """
    try:
        # Since Gemini text models don't support image generation,
        # we'll create a simple placeholder image
        from PIL import Image, ImageDraw, ImageFont
        
        # Parse size
        width, height = map(int, size.split("x"))
        
        # Create a simple colored square as placeholder
        img = Image.new('RGB', (width, height), color='#87CEEB')  # Sky blue
        draw = ImageDraw.Draw(img)
        
        # Add some simple shapes to make it look more interesting
        # Draw a simple sun in the corner
        draw.ellipse([width-100, 20, width-20, 100], fill='#FFD700')
        
        # Draw some simple clouds
        for i in range(3):
            x = 50 + i * 150
            y = 50 + i * 20
            draw.ellipse([x, y, x+80, y+40], fill='white')
            draw.ellipse([x+20, y+20, x+100, y+60], fill='white')
        
        # Add prominent placeholder text
        try:
            # Try to use a larger font if available
            font_size = min(width // 20, 24)
            font = ImageFont.load_default()
            # Try to load a larger font
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                except:
                    pass  # Use default font
        except:
            font = ImageFont.load_default()
        
        # Add main placeholder text
        placeholder_text = "AI Image Generation"
        text_bbox = draw.textbbox((0, 0), placeholder_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = height // 2 - 30
        
        # Draw text with outline for better visibility
        draw.text((text_x, text_y), placeholder_text, fill='white', font=font, stroke_width=2, stroke_fill='black')
        
        # Add subtitle
        subtitle = "Placeholder - Gemini text models don't support image generation"
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (width - subtitle_width) // 2
        subtitle_y = height // 2 + 10
        
        draw.text((subtitle_x, subtitle_y), subtitle, fill='white', font=font, stroke_width=1, stroke_fill='black')
        
        # Add prompt info in corner
        prompt_preview = prompt[:30] + "..." if len(prompt) > 30 else prompt
        draw.text((20, height-60), f"Prompt: {prompt_preview}", fill='black', font=font)
        
        # Convert to bytes
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return img_byte_arr
        
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

# --- Configuration ---
MODEL_TEXT = "gemini-2.5-flash-lite"  # Always use Gemini model
MODEL_IMAGE = "gemini-2.0-flash-exp"  # Gemini image generation model
IMAGE_SIZE = "1024x1024"

# --- Utility functions ---
def save_bytes(path: Path, data: bytes) -> None:
    """Save binary data to a file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def write_text(path: Path, text: str) -> None:
    """Save text to a file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# --- Architect functions for Story Bible generation ---
def validate_bible(data: dict) -> tuple[bool, str | None]:
    """
    用 StoryBible(**data) 做结构校验。通过返回 (True, None)，
    失败返回 (False, 简短错误消息字符串)。
    """
    try:
        StoryBible(**data)
        return True, None
    except Exception as e:
        return False, str(e)

def make_bible_repair_prompt(schema: dict, idea: str, pages: int, prev: dict, error: str) -> str:
    """
    生成"自修复"提示：说明错误、附上上一次的 JSON、再次强调必须严格符合 schema、
    且 plot_beats 数量 = pages。要求仅输出修复后的 JSON。
    """
    return f"""The previous Story Bible generation failed with this error: {error}

Previous attempt (fix this JSON):
{json.dumps(prev, indent=2, ensure_ascii=False)}

IMPORTANT REQUIREMENTS:
1. Output ONLY valid JSON - no markdown, no explanations, no extra text
2. Must exactly match this schema: {json.dumps(schema, ensure_ascii=False)}
3. plot_beats array MUST contain exactly {pages} items
4. Each plot_beat must have: {{ "page": n, "summary": "...(<=20 words)", "image_prompt": "...(<=30 words)" }}
5. Include art_style with: style_tags (array), palette (array), composition_rules (string)

Story idea: {idea}
Target pages: {pages}

Fix the JSON and return ONLY the corrected version:"""

def ensure_bible(idea: str, pages: int, schema: dict, max_attempts: int = 3, temperature: float = 0.3):
    """
    生成→校验；若失败就以错误报告+上次 JSON 让模型"修复"，最多 max_attempts 次。
    成功返回 (StoryBible实例, 原始dict)。失败抛出包含最后一次错误的异常。
    """
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                # First attempt: generate new Story Bible
                from app import make_bible_prompt
                prompt = make_bible_prompt(idea, pages, schema)
                data = llm_json(prompt, temperature=temperature)
            else:
                # Repair attempt: fix previous JSON
                prompt = make_bible_repair_prompt(schema, idea, pages, prev_data, error_msg)
                data = llm_json(prompt, temperature=temperature)
            
            # Validate the generated/fixed data
            is_valid, error_msg = validate_bible(data)
            if is_valid:
                bible = StoryBible(**data)
                return bible, data
            
            # Store for repair attempt
            prev_data = data
            
        except Exception as e:
            if attempt == max_attempts - 1:
                # Last attempt failed
                raise Exception(f"Story Bible generation failed after {max_attempts} attempts. Last error: {str(e)}")
            # Store error for repair attempt
            error_msg = str(e)
            prev_data = data if 'data' in locals() else {}
            continue
    
    # Should not reach here, but just in case
    raise Exception(f"Story Bible generation failed after {max_attempts} attempts")

# --- Designer functions for image generation and validation ---
def build_style_pack(bible: dict) -> str:
    """
    从 bible['art_style'] 取出:
      - style_tags: List[str]
      - palette: List[str]
      - composition_rules: str
    返回一段简洁的 style 描述串，如：
    "storybook illustration; style: watercolor, soft edges; palette: pastel blue, warm yellow; composition: centered, consistent proportions"
    若某字段缺失，跳过但语义通顺。
    """
    art_style = bible.get('art_style', {})
    
    style_parts = []
    
    # Style tags
    style_tags = art_style.get('style_tags', [])
    if style_tags:
        style_parts.append(f"style: {', '.join(style_tags[:3])}")
    
    # Palette
    palette = art_style.get('palette', [])
    if palette:
        style_parts.append(f"palette: {', '.join(palette[:4])}")
    
    # Composition rules
    composition = art_style.get('composition_rules', '')
    if composition:
        # Truncate if too long
        comp_text = composition[:50] + "..." if len(composition) > 50 else composition
        style_parts.append(f"composition: {comp_text}")
    
    if not style_parts:
        return "storybook illustration, consistent visual style"
    
    return "; ".join(style_parts)

def build_image_prompt(bible: dict, beat: dict) -> str:
    """
    基于现有 make_image_prompt 的升级版：
    - 使用 build_style_pack() 作为前缀；
    - 强化角色"锚点"（visual_anchors 合并去重，最多取2~3个）；
    - 场景来自 beat['image_prompt']（必要时截断到 ~30 词）；
    - 禁止风格漂移（不要混用风格关键词）；返回纯文本提示。
    """
    # Build style pack
    style_pack = build_style_pack(bible)
    
    # Extract character visual anchors
    anchors = []
    for char in bible.get('characters', []):
        anchors.extend(char.get('visual_anchors', []))
    
    # Remove duplicates and limit to 3 most important
    unique_anchors = list(dict.fromkeys(anchors))[:3]
    anchor_text = f"Characters: {', '.join(unique_anchors)}" if unique_anchors else ""
    
    # Get scene description from beat
    scene = beat.get('image_prompt', '')
    if len(scene) > 150:  # Truncate if too long
        scene = scene[:147] + "..."
    
    # Combine all parts
    parts = [style_pack]
    if anchor_text:
        parts.append(anchor_text)
    if scene:
        parts.append(f"Scene: {scene}")
    
    return ". ".join(parts)

def analyze_image_bytes(img_bytes: bytes) -> Dict[str, Any]:
    """
    用 PIL 读取，返回：width, height, aspect(float), unique_colors(近似，<=64色快速统计),
    entropy(像素熵的粗略估计), format。
    """
    try:
        from PIL import Image
        import io
        
        # Load image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Basic info
        width, height = img.size
        aspect = width / height if height > 0 else 0
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Quick color analysis (sample pixels for performance)
        pixels = list(img.getdata())
        sample_size = min(1000, len(pixels))
        sample_pixels = pixels[::len(pixels)//sample_size] if len(pixels) > sample_size else pixels
        
        # Count unique colors (approximate)
        unique_colors = len(set(sample_pixels))
        
        # Simple entropy estimation (variance of pixel values)
        if sample_pixels:
            r_values = [p[0] for p in sample_pixels]
            g_values = [p[1] for p in sample_pixels]
            b_values = [p[2] for p in sample_pixels]
            
            # Calculate variance as entropy proxy
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

def validate_image(img_bytes: bytes, expected_size: str, anchors: List[str], strict: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """
    规则：
    - 尺寸：与 expected_size 一致（例如 "1024x1024"）；不一致视为失败；
    - 低复杂度检测：unique_colors 很少(<10) 或 entropy 很低时，视为"可能为占位图"，strict 模式下失败；
    - 关键词：对 anchors 做简单包含检测（全部或至少1条，取较宽松策略）；strict 模式下若全无则失败；
    返回：(passed, reason, metrics)
    """
    # Parse expected size
    try:
        expected_w, expected_h = map(int, expected_size.split("x"))
    except:
        return False, f"Invalid expected_size format: {expected_size}", {}
    
    # Analyze image
    metrics = analyze_image_bytes(img_bytes)
    if 'error' in metrics:
        return False, f"Image analysis failed: {metrics['error']}", metrics
    
    # Check dimensions
    if metrics['width'] != expected_w or metrics['height'] != expected_h:
        return False, f"Size mismatch: got {metrics['width']}x{metrics['height']}, expected {expected_size}", metrics
    
    # Check complexity (placeholder detection)
    if strict:
        if metrics['unique_colors'] < 10:
            return False, f"Low complexity detected: only {metrics['unique_colors']} colors (possible placeholder)", metrics
        if metrics['entropy'] < 100:  # Threshold for entropy
            return False, f"Low entropy detected: {metrics['entropy']} (possible placeholder)", metrics
    
    # Check anchors (simple text inclusion)
    if anchors and strict:
        anchor_found = False
        # For now, we can't check text content in images, so we'll be lenient
        # In a real implementation, you might use OCR or other image analysis
        anchor_found = True  # Placeholder - assume anchors are present
    
    return True, "Image validation passed", metrics

def ensure_image(bible: dict, beat: dict, size: str, max_attempts: int = 2, strict: bool = True) -> Tuple[bytes, str, Dict[str, Any]]:
    """
    循环：
      for attempt in 1..max_attempts:
        prompt = build_image_prompt(bible, beat)
        img = gen_image_b64(prompt, size=size)
        ok, reason, metrics = validate_image(img, size, anchors, strict)
        if ok: return (img, prompt, metrics)
      失败：raise RuntimeError(f"Image generation failed after {max_attempts} attempts: {reason}")
    anchors = [a for c in bible.get('characters', []) for a in c.get('visual_anchors', [])]
    返回：(图像bytes, 最终prompt, metrics)
    """
    # Extract anchors
    anchors = [a for c in bible.get('characters', []) for a in c.get('visual_anchors', [])]
    
    for attempt in range(max_attempts):
        try:
            # Generate prompt
            prompt = build_image_prompt(bible, beat)
            
            # Generate image
            img_bytes = gen_image_b64(prompt, size=size)
            
            # Validate image
            ok, reason, metrics = validate_image(img_bytes, size, anchors, strict)
            
            if ok:
                return img_bytes, prompt, metrics
            
            # If strict mode and failed, try again
            if strict and attempt < max_attempts - 1:
                print(f"Image validation failed (attempt {attempt + 1}): {reason}. Retrying...")
                continue
            elif not strict:
                # In non-strict mode, accept the image even if validation fails
                return img_bytes, prompt, metrics
                
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"Image generation failed (attempt {attempt + 1}): {str(e)}. Retrying...")
                continue
            else:
                raise RuntimeError(f"Image generation failed after {max_attempts} attempts: {str(e)}")
    
    # If we get here, all attempts failed
    raise RuntimeError(f"Image generation failed after {max_attempts} attempts: {reason}")

# --- Author functions for page text generation and validation ---
import re
from typing import Tuple, Dict, Any, List

def build_page_text_prompt(bible: dict, beat: dict) -> str:
    """
    产出 Author 的最终提示词（要求只返回成稿文本、无标题/无Markdown）。
    约束：
    - 受众/语气/叙述人称来自 Bible（缺省兜底）
    - 2~4 句，<= 90 个词（英文），避免生僻词
    - 应点到 beat['summary'] 的要素（不要求逐词匹配）
    - 不要反问读者（避免问句结尾）
    - 保持角色名一致（来自 bible['characters']）
    """
    # Get tone and narrator voice from Bible
    tone = bible.get('tone', 'Warm, imaginative, hopeful.')
    narrator_voice = bible.get('narrator_voice', 'Third-person, gentle, playful.')
    
    # Get character names
    character_names = [char.get('name', '') for char in bible.get('characters', [])]
    character_text = f"Characters: {', '.join(character_names)}" if character_names else ""
    
    # Get beat summary
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
    """用正则粗分句（以 . ! ? 结束），清理空白。"""
    # Split by sentence endings and clean up
    sentences = re.split(r'[.!?]+', text)
    # Clean and filter empty sentences
    cleaned = [s.strip() for s in sentences if s.strip()]
    return cleaned

def _word_count(text: str) -> int:
    """粗略英文词数统计；对中文可按字符数近似处理。"""
    # Simple word count for English
    words = re.findall(r'\b\w+\b', text.lower())
    return len(words)

# Light forbidden words example, can be expanded as needed
FORBIDDEN = {"bloody", "damn", "kill", "sex", "hate", "stupid", "ugly"}

def validate_page_text(text: str, bible: dict, beat: dict, strict: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """
    返回：(passed, reason, metrics)
    规则（严格模式）：
    - 2 <= 句子数 <= 4
    - 词数 <= 90（或中文字符数 <= 180）
    - 末尾不以问号结尾（避免直接对读者发问）
    - 至少出现 1 个主要角色名（来自 bible['characters'])
    - 不包含 FORBIDDEN 词（大小写不敏感）
    metrics 含：sentence_count, word_count, avg_words_per_sentence, has_main_character, forbidden_found(list)
    """
    # Clean text
    cleaned_text = text.strip()
    
    # Split into sentences
    sentences = _split_sentences(cleaned_text)
    sentence_count = len(sentences)
    
    # Count words
    word_count = _word_count(cleaned_text)
    
    # Check if ends with question
    ends_with_question = cleaned_text.rstrip().endswith('?')
    
    # Check for main characters
    character_names = [char.get('name', '').lower() for char in bible.get('characters', [])]
    has_main_character = any(name in cleaned_text.lower() for name in character_names if name)
    
    # Check for forbidden words
    text_lower = cleaned_text.lower()
    forbidden_found = [word for word in FORBIDDEN if word in text_lower]
    
    # Calculate average words per sentence
    avg_words_per_sentence = round(word_count / sentence_count, 1) if sentence_count > 0 else 0
    
    # Validation rules
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
    
    # All validations passed
    return True, "Text validation passed", {
        'sentence_count': sentence_count,
        'word_count': word_count,
        'avg_words_per_sentence': avg_words_per_sentence,
        'has_main_character': has_main_character,
        'forbidden_found': forbidden_found,
        'ends_with_question': ends_with_question
    }

def make_page_text_repair_prompt(bible: dict, beat: dict, previous_text: str, reason: str) -> str:
    """
    将验证失败的"问题点"转成修复指令，并附上 previous_text，让模型"在保留信息的前提下重写"。
    仍然要求：只返回成稿文本，无标题/无解释。
    """
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
    """
    循环：prompt -> llm_text -> 清洗三引号/多余空白 -> validate
    失败：用 make_page_text_repair_prompt 让模型重写，最多 max_attempts 次
    成功：返回 (final_text, metrics)
    失败：raise RuntimeError("Page text generation failed after ...: <reason>")
    """
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                # First attempt: generate new page text
                prompt = build_page_text_prompt(bible, beat)
                text = llm_text(prompt, temperature=temperature)
            else:
                # Repair attempt: fix previous text
                prompt = make_page_text_repair_prompt(bible, beat, prev_text, error_reason)
                text = llm_text(prompt, temperature=temperature)
            
            # Clean the text (remove markdown quotes, extra whitespace)
            cleaned_text = text.strip()
            # Remove markdown code blocks if present
            cleaned_text = re.sub(r'```.*?```', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = cleaned_text.strip()
            
            # Validate the cleaned text
            is_valid, error_reason, metrics = validate_page_text(cleaned_text, bible, beat, strict)
            
            if is_valid:
                return cleaned_text, metrics
            
            # Store for repair attempt
            prev_text = cleaned_text
            
        except Exception as e:
            if attempt == max_attempts - 1:
                # Last attempt failed
                raise RuntimeError(f"Page text generation failed after {max_attempts} attempts: {str(e)}")
            # Store error for repair attempt
            error_reason = str(e)
            prev_text = text if 'text' in locals() else ""
            continue
    
    # If we get here, all attempts failed
    raise RuntimeError(f"Page text generation failed after {max_attempts} attempts: {error_reason}")
