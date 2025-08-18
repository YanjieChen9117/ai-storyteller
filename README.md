# AI Storyteller

An AI-powered storybook generator that creates illustrated stories from a single idea. Built with Streamlit and Google Gemini AI.

## 🚀 Features

- **AI Story Generation**: Create complete storybooks with AI-generated text and images
- **Story Bible Creation**: Generate structured story outlines with characters, themes, and plot beats
- **Image Generation**: Create illustrations for each page using Gemini AI
- **PDF Export**: Download your completed storybook as a PDF
- **Student Learning**: Perfect for learning prompt engineering and AI workflow design

**Note**: Image generation currently uses placeholder images since Gemini text models don't support image generation. For production use, you can integrate with DALL-E, Midjourney, or Stable Diffusion.

## 🎨 Image Generation Status

**Current Limitation**: 
- Gemini 2.5 Flash Lite (text model) cannot generate images
- Gemini 2.0 Flash Exp is for image understanding, not generation
- Current implementation shows clear placeholder images with explanatory text

**Alternative Solutions**:
1. **OpenAI DALL-E**: Add `openai` package and configure `OPENAI_API_KEY`
2. **Stable Diffusion**: Use local or cloud-based Stable Diffusion API
3. **Midjourney**: Integrate via Discord bot API
4. **Custom Models**: Deploy your own image generation model

**Placeholder Images Include**:
- Clear "AI Image Generation" label
- Explanation of the limitation
- Original prompt preview
- Professional appearance for development/testing

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Text**: Google Gemini 2.5 Flash Lite
- **AI Images**: Google Gemini 2.0 Flash Exp
- **Backend**: Python 3.12+
- **Data Validation**: Pydantic
- **PDF Generation**: FPDF2

## 📋 Prerequisites

- Python 3.12 or higher
- Google Gemini API key
- Internet connection

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <https://github.com/SquareRoot49/ai-storyteller.git>
cd ai-storyteller
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
Create a `.env` file in the project root:
```bash
# Required: Your Google Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here
```

**Note**: Model configuration is now hardcoded in `utils.py`:
- Text generation: `gemini-2.5-flash-lite`
- Image generation: `gemini-2.0-flash-exp`
- Image size: `1024x1024`

### 5. Test Setup
```bash
python test_setup.py
```

### 6. Run the Application
```bash
streamlit run app.py
```

## 🎯 How It Works

### 1. **The Architect** (Story Bible Generation)
- Takes your story idea and target page count
- Generates a complete story structure using Gemini AI
- Creates character profiles, themes, and plot beats
- Ensures consistency across the entire story

### 2. **The Author** (Page Text Generation)
- Writes engaging text for each page
- Maintains consistent tone and voice
- Incorporates character development and plot progression

### 3. **The Designer** (Image Generation)
- Creates illustrations for each page using Gemini AI
- Maintains visual consistency across characters and settings
- Generates high-quality images that match the story content

### 4. **The Publisher** (Export & Distribution)
- Combines text and images into a cohesive storybook
- Exports to PDF format for easy sharing
- Creates ZIP archives for further editing

## 🎨 Student Learning Objectives

### **Your Role on the Team:**
- **The Architect**: Engineer prompts to create structured story plans
- **The Character & Style Designer**: Define the book's unique visual identity  
- **The Author**: Craft prompts to guide the AI's narrative voice
- **The Brand Manager**: Give your final project a catchy, creative name!

### **What You'll Learn:**
- Advanced Prompt Engineering
- Agentic Thinking & Workflows
- Creative AI Control & Consistency
- AI-powered Storytelling Techniques

## 🔧 Customization

### Prompt Engineering
Modify the prompt templates in `app.py`:
- `make_bible_prompt()`: Story structure generation
- `make_page_text_prompt()`: Page content creation
- `make_image_prompt()`: Image generation guidance

### Story Bible Schema
Extend the `StoryBible` class in `utils.py` to add:
- Additional character attributes
- New story elements
- Custom validation rules

### API Configuration
Adjust AI parameters in `utils.py`:
- Temperature settings for creativity control
- Token limits for response length
- Retry logic for reliability

## 📁 Project Structure

```
storyteller/
├── app.py              # Main Streamlit application
├── utils.py            # AI API wrappers and data models
├── test_setup.py       # Setup verification script
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
├── README.md          # This file
└── outputs/           # Generated storybooks (created at runtime)
    ├── story_name/
    │   ├── bible.json
    │   ├── images/
    │   └── pages/
```

## 🐛 Troubleshooting

### Common Issues

**API Key Errors**
- Ensure `GEMINI_API_KEY` is set in your `.env` file
- Verify your API key is valid and has sufficient quota

**Model Errors**
- Models are hardcoded in `utils.py` - no configuration needed
- Ensure you have access to Gemini 2.5 Flash Lite and 2.0 Flash Exp

**Image Generation Issues**
- Check if your Gemini API key supports image generation
- Verify network connectivity

**Dependency Issues**
- Run `pip install -r requirements.txt` to install all packages
- Ensure you're using Python 3.12+

### Getting Help

1. Run `python test_setup.py` to diagnose issues
2. Check the error messages in the Streamlit interface
3. Verify your `.env` file configuration
4. Ensure all dependencies are installed

## 📚 API Reference

### Core Functions

#### `llm_text(prompt, temperature=0.2, model=None, max_tokens=256)`
Generate text responses using Gemini AI.

#### `llm_json(prompt, temperature=0.2, model=None, max_tokens=2048)`
Generate structured JSON responses using Gemini AI.

#### `gen_image_b64(prompt, model="gemini-2.0-flash-exp", size="1024x1024")`
Generate images using Gemini AI.

### Data Models

#### `StoryBible`
Complete story specification with metadata, characters, and plot structure.

#### `Character`
Character definition with name, role, personality, and visual anchors.

#### `PlotBeat`
Individual page/scene specification with summary and image requirements.

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:
- Experiment with different prompt strategies
- Add new story elements and character types
- Improve the UI/UX design
- Share your creative story examples

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Google Gemini AI for providing the AI capabilities
- Streamlit for the web application framework
- The open source community for the supporting libraries

---

**Happy Storytelling! 📖✨**
