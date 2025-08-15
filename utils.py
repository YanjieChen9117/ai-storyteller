import os, base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# --- Models from env ---
MODEL_TEXT  = os.getenv("MODEL_TEXT",  "gpt-4o-mini")
MODEL_IMAGE = os.getenv("MODEL_IMAGE", "dall-e-3")
IMAGE_SIZE  = os.getenv("IMAGE_SIZE",  "1024x1024")

# Initialize OpenAI client with error checking
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key. See env_template.txt for an example.")

client = OpenAI(api_key=api_key)

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

# --- Robust API wrappers (students bring prompts & orchestration) ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def llm_text(prompt: str,
             model: str = MODEL_TEXT,
             response_format: Optional[Dict[str, Any]] = None,
             temperature: float = 0.7) -> str:
    """
    Single-shot text call with automatic retries. Students provide the full prompt.
    
    Args:
        prompt: The complete prompt (system/user style OK)
        model: OpenAI model to use
        response_format: For JSON output, pass {'type':'json_object'} or a json_schema dict
        temperature: Creativity level (0.0 = deterministic, 1.0 = very creative)
    
    Returns:
        The model's text response
        
    Raises:
        Exception: If API call fails after retries
    """
    try:
        # Use chat completions API with proper message format
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(model=model, messages=messages)
        if response_format:
            kwargs["response_format"] = response_format
        if temperature is not None:
            kwargs["temperature"] = temperature
        
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        raise Exception(f"LLM API call failed: {str(e)}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def gen_image_b64(prompt: str,
                  model: str = MODEL_IMAGE,
                  size: str = IMAGE_SIZE) -> bytes:
    """
    Generate an image from a text prompt with automatic retries.
    
    Args:
        prompt: Detailed description of the image to generate
        model: OpenAI image model to use
        size: Image dimensions (e.g., "1024x1024", "1792x1024")
    
    Returns:
        Image data as bytes (PNG format)
        
    Raises:
        Exception: If API call fails after retries
    """
    try:
        resp = client.images.generate(model=model, prompt=prompt, size=size)
        return base64.b64decode(resp.data[0].b64_json)
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")


def save_bytes(path: Path, data: bytes) -> None:
    """Save binary data to a file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_text(path: Path, text: str) -> None:
    """Save text to a file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
