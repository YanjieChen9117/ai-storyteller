# AI Storyteller — Starter (Part 1)

Turn a one-line idea into a multi-page, illustrated storybook with consistent plot and art style. This starter is intentionally simple: thin API wrappers, a single Streamlit page, and a strongly-typed Story Bible so beginners can succeed and advanced students can add "agentic" loops.

## 🎯 Project Overview

**The Mission**: Design the 'brain' of an AI agent that takes a simple idea and turns it into a fully illustrated, multi-page digital storybook with a consistent plot and art style.

**Your Role on the Team**:
- **The Architect**: Engineer prompts to create structured story plans
- **The Character & Style Designer**: Define the book's unique visual identity  
- **The Author**: Craft prompts to guide the AI's narrative voice
- **The Brand Manager**: Give your final project a catchy, creative name!

**What You'll Learn**:
- Advanced Prompt Engineering
- Agentic Thinking & Workflows
- Creative AI Control & Consistency

## ✨ Features
- **Single-pass baseline:** Idea → Story Bible (JSON) → Page Text → Image Prompts → Images → Export (PDF/ZIP).
- **Story Bible schema:** Characters, tone, style tags, palette, continuity rules, per-page plot beats.
- **All local outputs:** `outputs/<project>/bible.json`, `pages/`, `images/`, `book.pdf`, `story_export.zip`.
- **Thin wrappers:** Only two primitives to learn: `llm_text()` and `gen_image_b64()`.
- **Robust error handling:** Automatic retries and helpful error messages.

## 🛠️ Tech Stack
- **Python**, **Streamlit**
- **OpenAI API** (text via Responses API, images via Images API)
- **Pydantic** (schema/validation)
- **Pillow**, **fpdf2** (images/PDF)
- **python-dotenv**, **tenacity** (env/retries)

## 🚀 Quick Start

### Prerequisites
- Python **3.11+**
- An OpenAI API key

### Getting API Keys
1. Create/sign in to an [OpenAI account](https://platform.openai.com/)
2. Navigate to [API Keys](https://platform.openai.com/api-keys)
3. Create a new API key and keep it secret
4. Put it in a `.env` file (see below)

### Install & Run
```bash
# Clone or download this project
cd storyteller

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your API key

# Test your setup (optional but recommended)
python test_setup.py

# Run the app
streamlit run app.py
```

### Environment Setup
Create a `.env` file in the project root:
```bash
# Required: Your OpenAI API key
OPENAI_API_KEY=sk-your-api-key-here

# Optional: Model configuration
MODEL_TEXT=gpt-4o-mini
MODEL_IMAGE=dall-e-3
IMAGE_SIZE=1024x1024
```

## 📚 Student Tasks

1. **Modify the prompt templates** in `app.py`:
   - Improve `make_bible_prompt()` for better story structure
   - Enhance `make_page_text_prompt()` for better writing
   - Refine `make_image_prompt()` for consistent art style

2. **Add basic controls**:
   - Genre selection (fantasy, sci-fi, mystery, etc.)
   - Target age group
   - Story length preferences

3. **Implement agentic loops**:
   - Story revision based on feedback
   - Character consistency checks
   - Plot coherence validation
   - Style consistency enforcement

4. **Add creative features**:
   - Multiple art style options
   - Character customization
   - Plot twist generation
   - Alternative endings

## 🎨 Example Story Ideas

Try these to get started:
- "A robot learns to paint with emotions"
- "A magical library where books come to life"
- "A young detective solves mysteries with their pet dragon"
- "A time-traveling chef brings forgotten recipes back to life"
- "A shy cloud learns to make friends by creating beautiful rainbows"


## 🔧 Troubleshooting

### Common Issues

**"OPENAI_API_KEY not found"**
- Make sure you have a `.env` file in the project root
- Check that your API key is correct and has credits

**"API call failed"**
- Verify your internet connection
- Check if you have sufficient API credits
- Try simplifying your story idea

**"JSON parsing error"**
- The AI might have generated invalid JSON
- Try running again with a simpler prompt
- Check the console for detailed error messages

**"Image generation failed"**
- DALL-E 3 has content filters
- Avoid potentially problematic content
- Try rephrasing your image prompts

### Getting Help
- Check the [OpenAI API documentation](https://platform.openai.com/docs)
- Review the [Streamlit documentation](https://docs.streamlit.io/)
- Look at the error messages in the terminal for debugging clues

## 📁 Project Structure
```
storyteller/
├── app.py              # Main Streamlit application
├── utils.py            # API wrappers and Story Bible schema
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .env.example       # Environment template
├── test_setup.py      # Setup verification script
└── outputs/           # Generated stories (created automatically)
    └── <project_name>/
        ├── bible.json
        ├── pages/
        ├── images/
        ├── book.pdf
        └── story_export.zip
```

## 🎓 Learning Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Tutorial](https://docs.streamlit.io/library/get-started)
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**Happy storytelling! 🚀📖**
