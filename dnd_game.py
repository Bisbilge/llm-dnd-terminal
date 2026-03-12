#!/usr/bin/env python3
"""
DnD Terminal Game - Local LLM powered (completion-style)
Kullanım: python dnd_game.py -m /path/to/model.gguf
"""

import subprocess
import argparse
import os
import sys
import textwrap
import random
import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ── Renkler ──────────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GOLD    = "\033[33m"
    MAGENTA = "\033[95m"

# ── Hikaye Hafızası ───────────────────────────────────────────────────────────
class StoryMemory:
    """
    Tüm karakterlerin paylaştığı merkezi hikaye belleği.
    Her LLM çağrısına tutarlı bir bağlam sağlar.
    """
    def __init__(self, location: str, scenario_intro: str):
        self.location = location
        self.scenario_intro = scenario_intro
        self.events: List[Tuple[str, str]] = []   # (speaker, text)
        self.world_state: List[str] = []           # Sahnede bilinen olgular
        self.last_said: dict = {}                  # Her NPC'nin son sözü (tekrar önleme)
        self.turn: int = 0

    def add(self, speaker: str, text: str):
        self.events.append((speaker, text))
        # NPC'nin bu turdaki son sözünü kaydet (tekrar önleme için)
        if speaker != "DM":
            self.last_said[speaker] = text
        # Dünya durumunu güncelle (DM konuşmalarından)
        if speaker == "DM":
            self._extract_world_facts(text)

    def _extract_world_facts(self, dm_text: str):
        """
        DM metninden sahne olgularını çıkar.
        Ham cümleleri değil, anahtar olayları kısa etiket olarak sakla,
        böylece NPC'lerin prompt'larına DM metni birebir sızmaz.
        """
        if len(self.world_state) >= 6:
            return
        # Olgu anahtar kelimeleri: büyük harfle başlayan isimler, önemli eylemler
        # Basit heuristik: ilk cümleyi 60 karaktere kısalt, DM konuşma izi bırakmaz
        sentences = re.split(r'(?<=[.!?])\s+', dm_text.strip())
        for s in sentences[:2]:
            fact = s.strip()[:80]
            if fact and fact not in self.world_state:
                self.world_state.append(fact)
                break  # Her DM konuşmasından sadece 1 olgu al

    def narrative_context(self, last_n: int = 6) -> str:
        """Son N olaydan özlü bir bağlam metni üret."""
        if not self.events:
            return "The adventure is just beginning."
        recent = self.events[-last_n:]
        lines = []
        for speaker, text in recent:
            short = text[:100].rstrip()
            if len(text) > 100:
                short += "..."
            lines.append(f"{speaker}: {short}")
        return " | ".join(lines)

    def world_summary(self) -> str:
        """Sahnedeki bilinen olguların özeti."""
        if not self.world_state:
            return ""
        return "Known facts: " + "; ".join(self.world_state[-4:])

    def party_context(self, party_chars) -> str:
        """Parti durumunu özetler."""
        members = ", ".join(
            f"{c.name} the {c.race} {c.cls} (HP:{c.hp}/{c.max_hp})"
            for c in party_chars
        )
        return f"Party: {members}."


# ── LLM ──────────────────────────────────────────────────────────────────────
LOG_PREFIXES = (
    'llama_', 'ggml_', 'main:', 'build:', 'warning:',
    'log_', 'INFO', 'load_', 'sampling', 'eval time', 'total time',
    'system_info', 'perf_', 'slot ', 'srv ', 'clip_',
)

# Model completion'ı koparması gereken meta-yazı kalıpları
STOP_PHRASES = (
    "How do", "Provide a", "says (one", "- Detect",
    "Choose", "Options", "What do you do", "Action:",
    "Roll for", "Initiative", "The party decides",
    "Player:", "User:", "Human:",
)

def _clean_llama_output(raw: str, prompt: str, end_on_quote: bool = True) -> str:
    """Ham llama-cli çıktısını temizler ve döndürür."""
    output = raw.strip()

    # Prompt kısmını çıkar
    prompt_stripped = prompt.strip()
    if prompt_stripped in output:
        output = output[output.index(prompt_stripped) + len(prompt_stripped):]

    # Log satırlarını temizle
    clean_lines = []
    for line in output.split('\n'):
        if any(line.lstrip().startswith(p) for p in LOG_PREFIXES):
            continue
        clean_lines.append(line)
    cleaned = '\n'.join(clean_lines).strip()

    # | beam search artifactı — SADECE diyalog modunda kes (seçeneklerde değil)
    if end_on_quote and '|' in cleaned:
        cleaned = cleaned.split('|')[0].strip()

    # Kapanış tırnağında kes (diyalog tamamlandı)
    if end_on_quote and prompt_stripped.endswith('"') and '"' in cleaned:
        cleaned = cleaned.split('"')[0].strip()

    # Meta-yazı sızıntılarını temizle
    for phrase in STOP_PHRASES:
        if phrase in cleaned:
            cleaned = cleaned[:cleaned.index(phrase)].strip()

    return cleaned.strip()


def _run_llama(model_path: str, prompt: str, max_tokens: int,
               temp: float = 0.85, max_sentences: int = 3) -> str:
    """
    llama-cli çalıştır, çıktıyı temizle, max_sentences cümle döndür.
    max_sentences=0 → tüm metni döndür (seçenekler için).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    binary = os.path.join(script_dir, "build", "bin", "llama-cli")

    cmd = [
        binary,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temp),
        "-c", "2048",
        "-s", str(random.randint(1, 9999)),
        "-b", "1",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=120
        )
        cleaned = _clean_llama_output(result.stdout, prompt)

        if max_sentences > 0:
            sentences = re.split(r'(?<=[.!?])\s+', cleaned)
            # Sadece tam biten cümleleri al (. ! ? ile biten)
            complete = [s for s in sentences if re.search(r'[.!?]$', s.strip())]
            chosen = complete[:max_sentences] if complete else sentences[:max_sentences]
            result_text = ' '.join(chosen).strip()
            if not result_text:
                # Hiç tam cümle yoksa, mevcut metni son noktalamada kes
                m = re.search(r'^(.*[.!?])', cleaned, re.DOTALL)
                result_text = m.group(1).strip() if m else cleaned[:280].strip()
            # İlk harfi büyüt
            if result_text:
                result_text = result_text[0].upper() + result_text[1:]
            return result_text
        else:
            return cleaned

    except subprocess.TimeoutExpired:
        return "[yanıt zaman aşımı]"
    except Exception as e:
        return f"[hata: {e}]"


def call_llm_dm(model_path: str, memory: StoryMemory, party, player_action: str) -> str:
    """
    DM yanıtı — tüm hikaye bağlamını içerir.
    Combat meta-dili (initiative, roll for) engellenir.
    """
    prompt = (
        f"You are a Dungeon Master narrating a fantasy story.\n"
        f"Location: {memory.location}.\n"
        f"{memory.world_summary()}\n"
        f"Story so far: {memory.narrative_context(6)}\n"
        f"{memory.party_context(party)}\n"
        f"Player action: {player_action}\n\n"
        f"Narrate what happens next in 2-3 sentences. "
        f"Stay consistent with prior events. "
        f"Be descriptive and atmospheric. "
        f"DO NOT say 'Roll for initiative' or other game mechanics. "
        f'Dungeon Master: "'
    )
    return _run_llama(model_path, prompt, max_tokens=180, temp=0.85, max_sentences=3)


def call_llm_npc(model_path: str, memory: StoryMemory, char: "Character",
                 triggering_event: str, other_npcs_said: List[Tuple[str, str]],
                 player_action: str = "") -> str:
    """
    NPC yanıtı.
    - Diğer NPC'lerin bu turdaki sözlerini görür (tekrar önleme)
    - Kendi önceki sözünü görür (aynı cümleyi tekrar söylemez)
    - Oyuncunun aksiyonunu görür (ilgili NPC'ye doğru tepki verir)
    """
    others_context = ""
    if other_npcs_said:
        # Her önceki NPC'nin sözünü satır satır göster, benzeri yasak
        lines = "\n".join(f'  - {sp} said: "{tx[:70]}"' for sp, tx in other_npcs_said)
        others_context = (
            f"Party members already spoke this turn:\n{lines}\n"
            f"You MUST say something DIFFERENT — different idea, different wording, different tone.\n"
        )

    # Kendi önceki turdan sözünü bildir
    prev_said = memory.last_said.get(char.name, "")
    avoid_repeat = ""
    if prev_said:
        avoid_repeat = (
            f"You said last turn: \"{prev_said[:70]}\"\n"
            f"Do NOT repeat or rephrase this. Say something new.\n"
        )

    # Oyuncu aksiyonu verilmişse NPC'nin buna göre tepki vermesini sağla
    action_context = ""
    if player_action:
        action_context = f"Player just did: {player_action[:100]}\n"

    prompt = (
        f"Fantasy adventure story. Location: {memory.location}.\n"
        f"Story so far: {memory.narrative_context(4)}\n"
        f"Just happened: {triggering_event[:150]}\n"
        f"{action_context}"
        f"{others_context}"
        f"{avoid_repeat}"
        f"{char.name} is a {char.race} {char.cls}. "
        f"Personality: {char.personality_tags}. "
        f"Speech style: {char.speech_style}\n"
        f"Write ONE short in-character reaction. Be specific to this moment, not generic.\n"
        f'{char.name} says: "'
    )
    return _run_llama(model_path, prompt, max_tokens=70, temp=0.85, max_sentences=2)


def call_llm_opening(model_path: str, memory: StoryMemory, opening_context: str) -> str:
    """
    Senaryo açılış sahnesi.
    Few-shot örnek ile modeli atmosferik betimlemeye yönlendirir,
    combat/initiative tetikleyicilerinden uzak tutar.
    """
    prompt = (
        f"You are a Dungeon Master narrating the opening of a fantasy adventure.\n"
        f"Location: {memory.location}.\n"
        f"Context: {opening_context}\n\n"
        f"Write an atmospheric, immersive opening scene description. "
        f"DO NOT start combat. DO NOT say 'Roll for initiative'. "
        f"Describe sights, sounds, smells and mood in 3-4 vivid sentences.\n\n"
        f"Example of good opening: "
        f'"The ancient forest closes around you like a held breath. '
        f'Pale witch-lights drift between gnarled oaks, casting long shadows on the mossy ground. '
        f'Somewhere deep in the dark, a child\'s voice whispers a lullaby with no tune you recognise."\n\n'
        f'The Dungeon Master opens the adventure: "'
    )
    return _run_llama(model_path, prompt, max_tokens=200, temp=0.88, max_sentences=4)


def _parse_options(raw: str) -> list:
    """
    Model çıktısından 4 aksiyon seçeneği çıkar.
    Desteklenen formatlar:
      - Satır bazlı:  "1. Foo\n2. Bar\n..."
      - Pipe bazlı:   "1. Foo | 2. Bar | ..."
      - Inline:       "1. Foo 2. Bar 3. Baz 4. Qux"
    """
    # | ve \n ile böl
    text = raw.replace('|', '\n').replace('\r', '\n')
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    options = []
    for line in lines:
        # Bir satırda inline numaralı maddeler varsa böl: "1. Foo 2. Bar ..."
        parts = re.split(r'(?=\b[2-9]\.\s)', line)
        for part in parts:
            part = part.strip()
            m = re.match(r'^(\d+[\.\)])\s*(.+)', part)
            if not m:
                continue
            text_part = m.group(2).strip()

            # Tırnak işaretlerini soy (model bazen 'action' veya "action" üretir)
            text_part = text_part.strip('"\'')

            # Seçenek içine sızan NPC diyaloğunu kes:
            # 'text. CharacterName:' veya "text.'Name:" kalıplarında kes
            npc_leak = re.search(r"[.!?'\"]?\s*[A-Z][a-z]+\s*:", text_part)
            if npc_leak:
                text_part = text_part[:npc_leak.start()].strip().rstrip(".'\"")

            # Çok kısa, boş veya tekrar eden seçenekleri atla
            if len(text_part) > 4 and text_part not in options:
                options.append(text_part)

    return options


def call_llm_options(model_path: str, memory: StoryMemory, player: "Character") -> list:
    """
    Oyuncu seçenekleri — mevcut sahneyle tutarlı 4 aksiyon önerisi.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    binary = os.path.join(script_dir, "build", "bin", "llama-cli")

    prompt = (
        f"Fantasy adventure. Location: {memory.location}.\n"
        f"Recent events: {memory.narrative_context(4)}\n"
        f"You are a {player.race} {player.cls} named {player.name}.\n\n"
        f"Write 4 short actions you could take right now. "
        f"First person singular (start each with 'I'). Under 8 words each. One per line.\n"
        f"Example format:\n"
        f"1. I draw my sword and look around.\n"
        f"2. I ask Thorin what he sees.\n"
        f"3. I move cautiously into the shadows.\n"
        f"4. I cast a detection spell.\n\n"
        f"Now write 4 actions for the current scene:\n"
        f"1. I "
    )

    cmd = [
        binary,
        "-m", model_path,
        "-p", prompt,
        "-n", "120",
        "--temp", "0.72",
        "-c", "2048",
        "-s", str(random.randint(1, 9999)),
        "-b", "1",
    ]

    default_opts = [
        "Etrafı dikkatlice araştırıyorum.",
        "Mevcut tehdide saldırıyorum.",
        "Savunma pozisyonu alıp bekliyorum.",
        "Güvenli bir yere doğru geri çekiliyorum."
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
        cleaned = _clean_llama_output(result.stdout, prompt, end_on_quote=False)
        raw = "1. I " + cleaned
        options = _parse_options(raw)

        # 3. şahıs sızıntısını düzelt: "b draws" → "I draw", "Player scans" → "I scan"
        fixed = []
        for opt in options:
            # "PlayerName verb..." → "I verb..."
            opt = re.sub(
                rf'^{re.escape(player.name)}\s+', 'I ', opt, flags=re.IGNORECASE
            )
            # Zaten "I " ile başlamıyorsa ekle (infinitive: "Draw sword" → "I draw sword")
            if not opt.lower().startswith("i ") and not opt.lower().startswith("i'"):
                opt = "I " + opt[0].lower() + opt[1:]
            # İlk harfi büyüt
            opt = opt[0].upper() + opt[1:]
            fixed.append(opt)
        options = fixed

        while len(options) < 4:
            options.append(default_opts[len(options)])

        return options[:4]

    except subprocess.TimeoutExpired:
        return default_opts
    except Exception as e:
        # Sessiz yutma yerine ekrana yaz (debug için)
        print(f"\n{C.DIM}  [options hata: {e}]{C.RESET}")
        return default_opts


# ── Karakterler ───────────────────────────────────────────────────────────────
@dataclass
class Character:
    name: str
    cls: str
    race: str
    hp: int
    max_hp: int
    ac: int
    color: str
    personality_tags: str
    speech_style: str
    is_player: bool = False

    def hp_bar(self):
        filled = int((self.hp / self.max_hp) * 10)
        bar = "█" * filled + "░" * (10 - filled)
        col = C.GREEN if self.hp > self.max_hp * 0.5 else C.YELLOW if self.hp > self.max_hp * 0.25 else C.RED
        return f"{col}{bar} {self.hp}/{self.max_hp}{C.RESET}"

    def status_line(self):
        star = f" {C.GOLD}★{C.RESET}" if self.is_player else ""
        return f"  {self.color}{C.BOLD}{self.name}{C.RESET} [{self.cls}] HP:{self.hp_bar()} AC:{self.ac}{star}"

PARTY = [
    Character("Thorin", "Fighter", "Dwarf", 28, 28, 18, C.RED,
              "gruff, honorable, battle-hardened, loyal to death",
              "Speaks in short blunt sentences. Sometimes uses dwarvish curses."),
    Character("Lyra", "Wizard", "Elf", 16, 16, 12, C.CYAN,
              "intellectual, arrogant, curious about magic, secretly fearful",
              "Speaks precisely and references arcane lore frequently."),
    Character("Brother Aldric", "Cleric", "Human", 22, 22, 16, C.GOLD,
              "devout but doubt-ridden, dark humor, fiercely protective",
              "Speaks with religious references and dry wit."),
]

# ── Senaryolar ────────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "name": "🌲 Fısıldayan Orman",
        "location": "Whispering Woods",
        "opening": (
            "An adventuring party arrives at the edge of a dark ancient forest. "
            "Strange lights flicker between the trees. Villagers have gone missing. "
            "Set the eerie opening scene:"
        )
    },
    {
        "name": "🏰 Terk Edilmiş Zindan",
        "location": "Grimhold Castle Dungeon",
        "opening": (
            "The party descends into crumbling dungeon ruins. "
            "Torchlight reveals ancient bones. Something moves in the darkness. "
            "Begin the adventure:"
        )
    },
    {
        "name": "🏚️ Gizemli Kasaba",
        "location": "Millhaven Town",
        "opening": (
            "The party arrives at a town with empty streets at midday. "
            "Shuttered windows, a distant church bell, smell of fear. "
            "Describe the unsettling scene:"
        )
    },
]

# ── UI ────────────────────────────────────────────────────────────────────────
def clear(): os.system('clear')

def header():
    print(f"""{C.GOLD}{C.BOLD}
╔══════════════════════════════════════════════════════════╗
║          ⚔️   DUNGEONS & DRAGONS  ⚔️                     ║
║              Powered by Local LLM                        ║
╚══════════════════════════════════════════════════════════╝
{C.RESET}""")

def show_party(party):
    print(f"\n{C.DIM}{'─'*62}{C.RESET}")
    for c in party:
        print(c.status_line())
    print(f"{C.DIM}{'─'*62}{C.RESET}\n")

def show_dm(text):
    print(f"\n{C.GOLD}{C.BOLD}📖 DM:{C.RESET}")
    print(textwrap.fill(text, 68, initial_indent="   ", subsequent_indent="   "))
    print()

def show_npc(char, text):
    print(f"{char.color}{C.BOLD}💬 {char.name}:{C.RESET}")
    print(textwrap.fill(text, 68, initial_indent="   ", subsequent_indent="   "))
    print()

def thinking(name):
    print(f"{C.DIM}  ⏳ {name} düşünüyor...{C.RESET}", end='\r', flush=True)

def clear_line():
    print(" " * 60, end='\r')

# ── Karakter Yaratma ──────────────────────────────────────────────────────────
def create_player() -> Character:
    print(f"\n{C.CYAN}{C.BOLD}✨ KARAKTERİNİ YARAT{C.RESET}\n")
    name = input("  Adın: ").strip() or "Kahraman"

    classes = ["Barbarian","Bard","Cleric","Druid","Fighter",
               "Monk","Paladin","Ranger","Rogue","Sorcerer","Warlock","Wizard"]
    races   = ["Human","Elf","Dwarf","Halfling","Half-Orc","Tiefling","Dragonborn"]

    print(f"\n  Sınıflar: {', '.join(classes)}")
    cls = input("  Sınıf: ").strip().capitalize()
    if cls not in classes: cls = "Fighter"

    print(f"\n  Irklar: {', '.join(races)}")
    race = input("  Irk: ").strip().capitalize()
    if race not in races: race = "Human"

    backstory = input("\n  Kısa geçmiş (boş bırakabilirsin): ").strip()
    personality = backstory or f"brave {race} {cls} seeking glory"

    hp_map = {"Barbarian":35,"Fighter":28,"Paladin":25,"Cleric":22,
              "Ranger":22,"Monk":20,"Druid":18,"Rogue":18,
              "Bard":16,"Warlock":16,"Sorcerer":14,"Wizard":14}
    hp = hp_map.get(cls, 20)

    return Character(
        name=name, cls=cls, race=race, hp=hp, max_hp=hp, ac=14,
        color=C.GREEN, personality_tags=personality,
        speech_style=f"Speaks as a {race} {cls} would.",
        is_player=True
    )

# ── Ana Döngü ─────────────────────────────────────────────────────────────────
def game_loop(model_path: str, player: Character, scenario: dict):
    party = PARTY + [player]

    # ── Merkezi hikaye belleği ──
    memory = StoryMemory(
        location=scenario["location"],
        scenario_intro=scenario["opening"]
    )

    clear()
    header()
    print(f"  {C.CYAN}Senaryo: {scenario['name']}  |  Lokasyon: {scenario['location']}{C.RESET}\n")

    # ── Açılış sahnesi ──
    thinking("DM")
    opening = call_llm_opening(model_path, memory, scenario["opening"])
    clear_line()
    show_dm(opening)
    memory.add("DM", opening)

    # NPC'ler açılışa tepki verirken birbirini duyuyor
    npcs_said_this_turn: List[Tuple[str, str]] = []
    for npc in PARTY:
        thinking(npc.name)
        resp = call_llm_npc(
            model_path, memory, npc,
            triggering_event=opening,
            other_npcs_said=npcs_said_this_turn,
            player_action=""
        )
        clear_line()
        show_npc(npc, resp)
        memory.add(npc.name, resp)
        npcs_said_this_turn.append((npc.name, resp))

    # ── Oyun döngüsü ──
    while True:
        thinking("Seçenekler hazırlanıyor")
        options = call_llm_options(model_path, memory, player)
        clear_line()
        print()  # thinking satırından sonra temiz boşluk

        show_party(party)
        print(f"{C.GREEN}{C.BOLD}⚔️  {player.name} ne yapıyor?{C.RESET}")
        for i, opt in enumerate(options, 1):
            print(f"  {C.CYAN}{i}.{C.RESET} {opt}")
        print(f"{C.DIM}\n  1-4 = Seç  |  kendi aksiyonunu yaz  |  çık = bitir{C.RESET}")

        action_input = input("\n  > ").strip()

        if not action_input:
            continue
        if action_input.lower() in ["çık", "quit", "exit", "q"]:
            print(f"\n{C.GOLD}Macera sona erdi. Güle güle!{C.RESET}\n")
            break

        # Seçim veya serbest yazı
        try:
            choice = int(action_input)
            action = options[choice - 1] if 1 <= choice <= len(options) else options[0]
        except ValueError:
            action = action_input

        memory.add(player.name, action)
        memory.turn += 1

        # ── DM tepkisi ──
        thinking("DM")
        dm_resp = call_llm_dm(model_path, memory, party, f"{player.name} decides to {action}")
        clear_line()
        show_dm(dm_resp)
        memory.add("DM", dm_resp)

        # ── 2 rastgele NPC — birbirinin bu turdaki sözünü ve oyuncunun aksiyonunu görüyor ──
        npcs_said_this_turn = []
        for npc in random.sample(PARTY, 2):
            thinking(npc.name)
            resp = call_llm_npc(
                model_path, memory, npc,
                triggering_event=dm_resp,
                other_npcs_said=npcs_said_this_turn,
                player_action=action
            )
            clear_line()
            show_npc(npc, resp)
            memory.add(npc.name, resp)
            npcs_said_this_turn.append((npc.name, resp))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model", required=True)
        args = parser.parse_args()

        if not os.path.exists(args.model):
            print(f"{C.RED}Model bulunamadı: {args.model}{C.RESET}")
            sys.exit(1)

        clear()
        header()

        player = create_player()

        print(f"\n{C.GOLD}{C.BOLD}SENARYO SEÇ:{C.RESET}")
        for i, s in enumerate(SCENARIOS, 1):
            print(f"  {i}. {s['name']}")
        try:
            scenario = SCENARIOS[int(input("\n  Seçim (1-3): ").strip()) - 1]
        except (ValueError, IndexError):
            scenario = SCENARIOS[0]

        game_loop(args.model, player, scenario)
    except (KeyboardInterrupt, EOFError):
        print(f"\n{C.GOLD}Macera sona erdi. Güle güle!{C.RESET}")
    finally:
        print(C.RESET, end="")

if __name__ == "__main__":
    main()