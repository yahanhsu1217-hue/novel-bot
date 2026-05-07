"""
同人 / 原創小說連載生成器
"""

from __future__ import annotations

import json
import os
import streamlit as st
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LENGTH_CHARS
from generator import stream_chapter, summarize_chapter, extract_story_bible, fix_consistency

LAST_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_settings.json")
LAST_SESSION_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_session.json")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="小說連載生成器",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { width: 370px !important }
  .block-container { padding-top: 1.6rem }
  .chapter-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px;
    padding: 28px 36px;
    font-size: 1.05rem;
    line-height: 2.1;
    white-space: pre-wrap;
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

# Auto-load last-used settings and session on fresh page open
if "_settings_initialized" not in st.session_state:
    st.session_state._settings_initialized = True
    # Restore settings
    try:
        with open(LAST_SETTINGS_FILE, encoding="utf-8") as _f:
            _last = json.load(_f)
        for _key in ["world_mode", "world_input", "character_notes", "language",
                     "perspective", "length_label", "total_chapters", "cp_type", "cp_character",
                     "intimacy", "name", "nickname", "gender", "personality", "appearance",
                     "residence", "background", "plot_want", "plot_forbid",
                     "environment", "nsfw"]:
            if _key in _last:
                st.session_state[f"w_{_key}"] = _last[_key]
        st.session_state["num_extra_chars"] = _last.get("num_extra_chars", 0)
        for _i, _c in enumerate(_last.get("extra_characters", [])):
            st.session_state[f"cn_{_i}"]   = _c.get("name", "")
            st.session_state[f"ca_{_i}"]   = _c.get("appearance", "")
            st.session_state[f"cp_{_i}"]   = _c.get("personality", "")
            st.session_state[f"cres_{_i}"] = _c.get("residence", "")
            st.session_state[f"csk_{_i}"]  = _c.get("skills", "")
            st.session_state[f"cr_{_i}"]   = _c.get("relationship", "")
    except Exception:
        pass
    # Restore chapters / story state
    try:
        with open(LAST_SESSION_FILE, encoding="utf-8") as _f:
            _sess = json.load(_f)
        st.session_state["chapters"]      = _sess.get("chapters", [])
        st.session_state["summaries"]     = _sess.get("summaries", [])
        st.session_state["story_bible"]   = _sess.get("story_bible", {"banned_phrases": [], "used_tropes": [], "open_threads": [], "established_facts": []})
        st.session_state["saved_settings"] = _sess.get("saved_settings", {})
        if st.session_state["chapters"]:
            st.session_state["story_started"] = True
    except Exception:
        pass

for k, v in [
    ("chapters", []),
    ("summaries", []),
    ("story_started", False),
    ("saved_settings", {}),
    ("num_extra_chars", 0),
    ("last_loaded_file", None),
    ("story_bible", {"banned_phrases": [], "used_tropes": [], "open_threads": [], "established_facts": []}),
    # Widget defaults
    ("w_world_mode",      "作品世界"),
    ("w_world_input",     ""),
    ("w_character_notes", ""),
    ("w_environment",     ""),
    ("w_language",        "繁體中文"),
    ("w_perspective",     "第一人稱（我）"),
    ("w_length_label",    "中篇（約 2000 字）"),
    ("w_total_chapters",  5),
    ("w_cp_type",         "無 CP"),
    ("w_cp_character",    ""),
    ("w_intimacy",        "糖（甜蜜互動）"),
    ("w_name",            ""),
    ("w_nickname",        ""),
    ("w_gender",          "女"),
    ("w_personality",     ""),
    ("w_appearance",      ""),
    ("w_residence",       ""),
    ("w_background",      ""),
    ("w_plot_want",       ""),
    ("w_plot_forbid",     ""),
    ("w_nsfw",            False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📖 故事設定")

    # ── Save / Load ───────────────────────────────────────────────────────────
    st.markdown("### 💾 儲存 / 載入設定")

    def _current_settings_json() -> str:
        chars = []
        for i in range(st.session_state.num_extra_chars):
            chars.append({
                "name":         st.session_state.get(f"cn_{i}", ""),
                "appearance":   st.session_state.get(f"ca_{i}", ""),
                "personality":  st.session_state.get(f"cp_{i}", ""),
                "residence":    st.session_state.get(f"cres_{i}", ""),
                "skills":       st.session_state.get(f"csk_{i}", ""),
                "relationship": st.session_state.get(f"cr_{i}", ""),
            })
        data = {
            "world_mode":       st.session_state.get("w_world_mode",       "作品世界"),
            "world_input":      st.session_state.get("w_world_input",       ""),
            "character_notes":  st.session_state.get("w_character_notes",  ""),
            "environment":      st.session_state.get("w_environment",       ""),
            "language":         st.session_state.get("w_language",          "繁體中文"),
            "perspective":      st.session_state.get("w_perspective",       "第一人稱（我）"),
            "length_label":     st.session_state.get("w_length_label",      "中篇（約 2000 字）"),
            "total_chapters":   st.session_state.get("w_total_chapters",    5),
            "cp_type":          st.session_state.get("w_cp_type",           "無 CP"),
            "cp_character":     st.session_state.get("w_cp_character",      ""),
            "intimacy":         st.session_state.get("w_intimacy",          "糖（甜蜜互動）"),
            "name":             st.session_state.get("w_name",              ""),
            "nickname":         st.session_state.get("w_nickname",          ""),
            "gender":           st.session_state.get("w_gender",            "女"),
            "personality":      st.session_state.get("w_personality",       ""),
            "appearance":       st.session_state.get("w_appearance",        ""),
            "residence":        st.session_state.get("w_residence",         ""),
            "background":       st.session_state.get("w_background",        ""),
            "plot_want":        st.session_state.get("w_plot_want",         ""),
            "plot_forbid":      st.session_state.get("w_plot_forbid",       ""),
            "nsfw":             st.session_state.get("w_nsfw",              False),
            "num_extra_chars":  st.session_state.num_extra_chars,
            "extra_characters": chars,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    st.download_button(
        label="⬇️ 儲存設定檔",
        data=_current_settings_json(),
        file_name="novel_settings.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded = st.file_uploader("載入設定檔（.json）", type="json",
                                label_visibility="collapsed")
    if uploaded is not None and uploaded.name != st.session_state.last_loaded_file:
        try:
            data = json.loads(uploaded.read().decode("utf-8"))
            for key in ["world_mode", "world_input", "character_notes", "language",
                        "perspective", "length_label", "total_chapters", "cp_type", "cp_character",
                        "intimacy", "name", "nickname", "gender", "personality", "appearance",
                        "residence", "background", "plot_want", "plot_forbid",
                        "environment", "nsfw"]:
                if key in data:
                    st.session_state[f"w_{key}"] = data[key]
            n = data.get("num_extra_chars", 0)
            st.session_state.num_extra_chars = n
            for i, c in enumerate(data.get("extra_characters", [])):
                st.session_state[f"cn_{i}"]   = c.get("name", "")
                st.session_state[f"ca_{i}"]   = c.get("appearance", "")
                st.session_state[f"cp_{i}"]   = c.get("personality", "")
                st.session_state[f"cres_{i}"] = c.get("residence", "")
                st.session_state[f"csk_{i}"]  = c.get("skills", "")
                st.session_state[f"cr_{i}"]   = c.get("relationship", "")
            st.session_state.last_loaded_file = uploaded.name
            st.rerun()
        except Exception as e:
            st.error(f"載入失敗：{e}")

    st.divider()

    # ── World setting ─────────────────────────────────────────────────────────
    st.markdown("### 🌍 世界設定")
    world_mode = st.radio(
        "類型", ["作品世界", "原創世界"],
        horizontal=True, label_visibility="collapsed", key="w_world_mode",
    )

    if world_mode == "作品世界":
        world_input = st.text_input(
            "作品名稱", key="w_world_input",
            placeholder="例：進擊的巨人、哈利波特、魷魚遊戲…",
        )
        character_notes = st.text_area(
            "原作角色外貌備注", key="w_character_notes",
            placeholder="例：\n萊納・布朗：金髮、藍眼、肌肉結實\n艾連・葉卡：黑髮、藍綠色眼睛、削瘦",
            height=100,
            help="AI 會嚴格以此為準，避免外貌描寫出錯",
        )
    else:
        world_input = st.text_area(
            "世界描述", key="w_world_input",
            placeholder="例：某天世界突然爆發僵屍病毒，城市淪陷，少數倖存者…\n或：我突然被選中進入一個孤島求生節目…",
            height=110,
        )
        character_notes = ""

    st.divider()

    # ── Environment ───────────────────────────────────────────────────────────
    st.markdown("### 🏙️ 環境設定")
    environment = st.text_area(
        "故事環境", key="w_environment",
        placeholder="例：廢棄的末日城市、古代皇宮、現代高中校園、太空站…\n可描述氛圍、時間點、特殊規則等",
        height=90,
    )

    st.divider()

    # ── Story settings ────────────────────────────────────────────────────────
    st.markdown("### ⚙️ 故事設定")
    language = st.radio(
        "語言", ["繁體中文", "簡體中文"],
        horizontal=True, key="w_language",
    )
    perspective = st.radio(
        "敘事視角", ["第一人稱（我）", "第三人稱（名字）"],
        horizontal=True, key="w_perspective",
        help="第一人稱：旁白全程用「我」；第三人稱：旁白用主角名字",
    )
    length_label = st.select_slider(
        "每章長度", options=list(LENGTH_CHARS.keys()),
        key="w_length_label",
    )
    total_chapters = st.number_input(
        "預計總章數", min_value=1, max_value=50,
        step=1, key="w_total_chapters",
        help="AI 會根據進度調整節奏，最後一章自動收尾",
    )

    st.divider()

    # ── CP ────────────────────────────────────────────────────────────────────
    st.markdown("### 💞 CP 設定")
    cp_type = st.radio(
        "配對", ["無 CP", "我 × 角色"],
        horizontal=True, label_visibility="collapsed", key="w_cp_type",
    )
    cp_character = ""
    intimacy = "糖（甜蜜互動）"
    if cp_type == "我 × 角色":
        cp_character = st.text_input(
            "角色名稱", key="w_cp_character",
            placeholder='輸入角色名，或輸入「隨機」由 AI 決定',
        )
        intimacy = st.select_slider(
            "親密程度",
            options=["清水（純愛暗戀）", "糖（甜蜜互動）", "甜虐（曖昧張力）", "熾熱（激情親密）"],
            key="w_intimacy",
        )

    st.divider()

    # ── NSFW ──────────────────────────────────────────────────────────────────
    st.markdown("### 🔞 限制級設定")
    nsfw = st.toggle("開啟限制級內容", key="w_nsfw",
                     help="開啟後允許生成明確的成人親密場景")
    if nsfw:
        st.warning("⚠️ 限制級模式已開啟", icon="🔞")

    st.divider()

    # ── Self introduction ─────────────────────────────────────────────────────
    st.markdown("### 🪪 我的設定")
    name        = st.text_input("姓名",    key="w_name",        placeholder="你在故事中的名字")
    nickname    = st.text_input("暱稱",    key="w_nickname",    placeholder="其他角色對你的稱呼（可留空）")
    gender      = st.radio("性別", ["女", "男", "不設定"], horizontal=True, key="w_gender")
    personality = st.text_area("個性",    key="w_personality", placeholder="例：外冷內熱、敏銳果斷…",        height=72)
    appearance  = st.text_area("外貌",    key="w_appearance",  placeholder="例：長黑髮、眼神銳利…",          height=72)
    residence   = st.text_input("住的地方", key="w_residence",  placeholder="例：廢棄工廠頂樓、學生宿舍302室…")
    background  = st.text_area("背景故事", key="w_background",  placeholder="例：記憶缺失的前特工…",           height=90)

    st.divider()

    # ── Extra characters ──────────────────────────────────────────────────────
    st.markdown("### 👥 額外角色設定")
    col_add, col_del = st.columns(2)
    with col_add:
        if st.button("＋ 新增角色", use_container_width=True):
            st.session_state.num_extra_chars += 1
    with col_del:
        if st.button("－ 移除角色", use_container_width=True,
                     disabled=st.session_state.num_extra_chars == 0):
            st.session_state.num_extra_chars -= 1

    extra_characters = []
    for i in range(st.session_state.num_extra_chars):
        with st.expander(f"角色 {i + 1}", expanded=True):
            c_name         = st.text_input("名稱",       key=f"cn_{i}", placeholder="角色名字")
            c_appearance   = st.text_input("外貌",       key=f"ca_{i}", placeholder="例：銀髮紅眼、高挑冷峻…")
            c_personality  = st.text_input("個性",       key=f"cp_{i}", placeholder="例：冷靜、腹黑…")
            c_residence    = st.text_input("住的地方",   key=f"cres_{i}", placeholder="例：城堡東翼、學校宿舍…")
            c_skills       = st.text_input("技能",       key=f"csk_{i}", placeholder="例：劍術、魔法、駭客…")
            c_relationship = st.text_input("與我的關係", key=f"cr_{i}", placeholder="例：青梅竹馬、宿敵…")
            extra_characters.append({
                "name": c_name, "appearance": c_appearance,
                "personality": c_personality, "residence": c_residence,
                "skills": c_skills, "relationship": c_relationship,
            })

    st.divider()

    # ── Plot settings ─────────────────────────────────────────────────────────
    st.markdown("### 📝 劇情設定")
    plot_want = st.text_area(
        "希望出現", key="w_plot_want",
        placeholder="例：與CP的誤會與和解、危機中展現能力…", height=90,
    )
    plot_forbid = st.text_area(
        "禁止出現", key="w_plot_forbid",
        placeholder="例：主角死亡、NTR、悲劇結局…", height=72,
    )

    st.divider()

    # ── Action buttons ────────────────────────────────────────────────────────
    start_btn = st.button("✨ 開始新故事", type="primary", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_settings() -> dict:
    return dict(
        world_mode=world_mode,
        world_input=world_input.strip(),
        language=language,
        perspective=perspective,
        length_label=length_label,
        cp_type=cp_type,
        cp_character=cp_character.strip(),
        name=name.strip(),
        nickname=nickname.strip(),
        gender=gender,
        personality=personality.strip(),
        appearance=appearance.strip(),
        background=background.strip(),
        extra_characters=extra_characters,
        plot_want=plot_want.strip(),
        plot_forbid=plot_forbid.strip(),
        total_chapters=int(total_chapters),
        character_notes=character_notes.strip(),
        intimacy=intimacy,
        environment=environment.strip(),
        residence=residence.strip(),
        nsfw=nsfw,
    )


def _save_session():
    try:
        data = {
            "chapters":      st.session_state.chapters,
            "summaries":     st.session_state.summaries,
            "story_bible":   st.session_state.story_bible,
            "saved_settings": st.session_state.saved_settings,
        }
        with open(LAST_SESSION_FILE, "w", encoding="utf-8") as _f:
            json.dump(data, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _validate(s: dict) -> list[str]:
    missing = []
    if not s["world_input"]:
        missing.append("作品名稱" if s["world_mode"] == "作品世界" else "世界描述")
    if not s["name"]:         missing.append("姓名")
    if not s["personality"]:  missing.append("個性")
    if not s["appearance"]:   missing.append("外貌")
    if not s["background"]:   missing.append("背景故事")
    if s["cp_type"] == "我 × 角色" and not s["cp_character"]:
        missing.append("配對角色名稱")
    return missing

# ── API client ────────────────────────────────────────────────────────────────

if not DEEPSEEK_API_KEY:
    st.error("❌ DEEPSEEK_API_KEY 未設定，請在 .env 檔案中加入 API 金鑰。")
    st.stop()

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ── Generate logic ────────────────────────────────────────────────────────────

def generate_chapter(settings: dict, chapter_num: int, prev_text: str = "", is_final: bool = False):
    full_text = ""
    placeholder = st.empty()
    for chunk in stream_chapter(
        client=client,
        chapter_num=chapter_num,
        prev_chapter_text=prev_text,
        is_final=is_final,
        prev_summaries=st.session_state.summaries,
        story_bible=st.session_state.story_bible,
        **settings,
    ):
        full_text += chunk
        placeholder.markdown(
            f'<div class="chapter-box">{full_text}</div>',
            unsafe_allow_html=True,
        )
    with st.spinner("🔍 校正內容一致性…"):
        protagonist_name = settings.get("name", "") if settings else ""
        extra_names = [c["name"] for c in (settings.get("extra_characters") or []) if c.get("name", "").strip()]
        full_text = fix_consistency(client, full_text, protagonist_name, extra_names, settings.get("nickname", ""))
        placeholder.markdown(
            f'<div class="chapter-box">{full_text}</div>',
            unsafe_allow_html=True,
        )
    with st.spinner("📝 記錄章節記憶…"):
        summary, bible_update = summarize_chapter(client, full_text, chapter_num)
    st.session_state.summaries.append(summary)
    # Merge into story bible
    b = st.session_state.story_bible
    b["banned_phrases"].extend(bible_update.get("banned_phrases", []))
    b["used_tropes"].extend(bible_update.get("used_tropes", []))
    b["open_threads"] = bible_update.get("open_threads", b["open_threads"])
    # Accumulate established facts (newer facts override older ones for same subject)
    b.setdefault("established_facts", []).extend(bible_update.get("established_facts", []))
    _save_session()
    return full_text

# ── Button handlers ───────────────────────────────────────────────────────────

if start_btn:
    s = _collect_settings()
    missing = _validate(s)
    if missing:
        st.warning(f"請填寫：{'、'.join(missing)}")
        st.stop()
    try:
        with open(LAST_SETTINGS_FILE, "w", encoding="utf-8") as _f:
            _f.write(_current_settings_json())
    except Exception:
        pass
    st.session_state.chapters = []
    st.session_state.summaries = []
    st.session_state.story_bible = {"banned_phrases": [], "used_tropes": [], "open_threads": [], "established_facts": []}
    st.session_state.saved_settings = s
    try:
        os.remove(LAST_SESSION_FILE)
    except FileNotFoundError:
        pass
    st.session_state.story_started = True
    is_final = s["total_chapters"] == 1
    chapter_text = generate_chapter(s, chapter_num=1, is_final=is_final)
    st.session_state.chapters.append(chapter_text)
    st.rerun()

# ── Main display ──────────────────────────────────────────────────────────────

st.markdown("## 📖 小說連載生成器")
st.caption("作品同人 × 原創世界 × 連載章節，把自己寫進故事裡")

if not st.session_state.chapters:
    st.markdown("""
<div style="text-align:center;padding:80px 20px;color:#475569">
  <div style="font-size:3.5rem">📖</div>
  <div style="font-size:1.15rem;margin-top:14px;font-weight:600">在左側填寫設定，開始你的故事</div>
  <div style="font-size:.92rem;margin-top:8px;color:#64748b">
    支援任何作品世界 · 原創設定 · 連載章節
  </div>
</div>
""", unsafe_allow_html=True)
else:
    s = st.session_state.saved_settings
    world_label = f"《{s['world_input']}》" if s["world_mode"] == "作品世界" else s["world_input"][:30] + "…"
    cp_label = f"♡ {s['cp_character']}" if s["cp_type"] == "我 × 角色" else "無 CP"
    st.markdown(f"**{world_label}**　·　{s['name']}　·　{s['language']}　·　{s['length_label']}　·　{cp_label}")
    st.caption(f"共 {len(st.session_state.chapters)} 章")
    st.divider()

    for i, text in enumerate(st.session_state.chapters):
        st.markdown(f"## 第 {i + 1} 章")
        st.markdown(f'<div class="chapter-box">{text}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Continue / Ending buttons ─────────────────────────────────────────────
    if st.session_state.story_started:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            continue_btn = st.button("➡️ 繼續下一章", type="primary", use_container_width=True)
        with btn_col2:
            ending_btn = st.button("🎬 寫結局", use_container_width=True)

        if continue_btn:
            s = st.session_state.saved_settings
            chapter_num = len(st.session_state.chapters) + 1
            is_final = chapter_num >= s["total_chapters"]
            chapter_text = generate_chapter(s, chapter_num=chapter_num,
                                            prev_text=st.session_state.chapters[-1], is_final=is_final)
            st.session_state.chapters.append(chapter_text)
            st.rerun()

        if ending_btn:
            s = st.session_state.saved_settings
            chapter_num = len(st.session_state.chapters) + 1
            chapter_text = generate_chapter(s, chapter_num=chapter_num,
                                            prev_text=st.session_state.chapters[-1], is_final=True)
            st.session_state.chapters.append(chapter_text)
            st.session_state.story_started = False
            st.rerun()

    st.divider()
    full_novel = "\n\n\n".join(st.session_state.chapters)
    title = s["world_input"][:20] if s["world_input"] else "小說"
    st.download_button(
        label=f"⬇️ 下載全部章節（{len(st.session_state.chapters)} 章）",
        data=full_novel,
        file_name=f"{title}_{s['name']}.txt",
        mime="text/plain",
    )
