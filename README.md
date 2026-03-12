# ⚔️ DnD Terminal — Local LLM Powered

A fully text-based Dungeons & Dragons adventure that runs entirely in your terminal, driven by a local Large Language Model. No internet. No API keys. No cloud.

The AI acts as your Dungeon Master — generating narrative, reacting to your choices, and giving each companion their own voice and memory of the story so far.

---

## Features

- **AI Dungeon Master** — Dynamic narration that adapts to every action you take
- **Living Companions** — Thorin, Lyra, and Brother Aldric each have distinct personalities and remember what's happened
- **Character Creation** — Pick your name, race, class, and backstory
- **Free-form Input** — Choose from 4 generated options *or* type any action you want
- **Fully Local** — All inference runs on your machine via `llama.cpp`

---

## Requirements

The game itself is pure Python with no dependencies. You need to supply two external components yourself (they are too large to include in the repo):

| Component | What to get |
|---|---|
| **Python** | 3.8 or higher |
| **A GGUF model** | Any quantized `.gguf` from Hugging Face — Llama 3, Mistral, or a roleplay-tuned model works great |
| **llama-cli binary** | Compiled from [llama.cpp](https://github.com/ggerganov/llama.cpp) or downloaded from its releases page |

> **Model recommendation:** For the best experience, use a model fine-tuned for roleplay or storytelling (search for `RP` or `story` tags on Hugging Face). Larger quants (Q4_K_M and above) produce noticeably better narrative quality.

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Bisbilge/llm-dnd-terminal.git
cd llm-dnd-terminal
```

**2. Place the `llama-cli` binary**

The game expects the binary at `build/bin/llama-cli` relative to the script. Create that directory and drop the binary in:
```bash
mkdir -p build/bin
cp /path/to/your/llama-cli build/bin/llama-cli
chmod +x build/bin/llama-cli
```

**3. Download a GGUF model**

You can store the `.gguf` file anywhere on your system. Just note the full path for when you run the game.

---

## Usage

```bash
python dnd_game.py -m /path/to/your/model.gguf
```

The game will walk you through character creation and scenario selection, then drop you into the adventure.

**At each turn:**
- Type `1`–`4` to pick a generated action
- Or type anything else to perform a custom action
- Type `q` or `exit` to end the session

---

## Performance Notes

Generation speed depends on your hardware and model size. Rough expectations:

| Hardware | Recommended model size |
|---|---|
| CPU only | 7B Q4 (slow but works) |
| 8 GB VRAM | 7B Q6 or 13B Q4 |
| 16 GB+ VRAM | 13B Q6 or 30B Q4 |

The game uses `llama-cli` in completion mode with a context of 2048 tokens. Each response generates 80–200 tokens, so turns should complete in under a minute on modern hardware.

---

## Project Structure

```
.
├── dnd_game.py        # Main game — all logic in one file
└── build/
    └── bin/
        └── llama-cli  # You provide this
```

---

## License

MIT
