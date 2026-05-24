"""
同人 / 原創小說連載生成器
"""

from __future__ import annotations

import json
import os
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, GEMINI_API_KEY, GEMINI_BASE_URL, LENGTH_CHARS
from generator import stream_chapter, stream_outline, summarize_chapter, extract_story_bible, fix_consistency, fix_sensory_crutches, fix_repetitive_paragraphs, fix_cross_chapter_repetition, extract_paragraph_starters, analyze_writing_style, STYLE_PRESETS, compress_old_summaries
from training import ISSUE_TYPES, add_note, delete_note, load_notes, notes_to_prompt_block

LAST_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_settings.json")
LAST_SESSION_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_session.json")

_DIRECTIVE_AS_ENDING_RULE = (
    "\n\n【⚠️ 本章以此指示作結 — 絕對執行，不可違背】\n"
    "以上指示描述的最後一個場景或事件，是本章的結束點。\n"
    "整章向此方向推進，寫完指示描述的內容後立即收章。\n"
    "嚴禁在指示事件結束後繼續加入新對話、新動作、新情節或後續發展。\n"
    "本章在指示的最後一刻自然結束，不留鉤子，不補充說明。"
)

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
                     "perspective", "length_label", "total_chapters", "cp_type",
                     "love_tone", "pacing",
                     "name", "nickname", "gender", "personality", "appearance",
                     "residence", "background", "protagonist_facts", "plot_want", "plot_forbid",
                     "environment", "nsfw", "style_reference"]:
            if _key in _last:
                st.session_state[f"w_{_key}"] = _last[_key]
        _cp_chars = _last.get("cp_characters", [])
        if not _cp_chars and _last.get("cp_character"):
            _cp_chars = [{"name": _last["cp_character"], "intimacy": _last.get("intimacy", "糖（甜蜜互動）")}]
        st.session_state["num_cp_chars"] = len(_cp_chars)
        for _j, _cp in enumerate(_cp_chars):
            st.session_state[f"cpname_{_j}"] = _cp.get("name", "")
            st.session_state[f"cpinti_{_j}"] = _cp.get("intimacy", "糖（甜蜜互動）")
            st.session_state[f"cpnotes_{_j}"] = _cp.get("notes", "")
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
        st.session_state["chapters"]              = _sess.get("chapters", [])
        st.session_state["summaries"]             = _sess.get("summaries", [])
        st.session_state["story_bible"]           = _sess.get("story_bible", {"banned_phrases": [], "used_tropes": [], "open_threads": [], "established_facts": [], "paragraph_starters": [], "asked_questions": []})
        st.session_state["saved_settings"]        = _sess.get("saved_settings", {})
        st.session_state["early_overview"]        = _sess.get("early_overview", "")
        st.session_state["early_overview_through"] = _sess.get("early_overview_through", 0)
        st.session_state["outlines"]              = _sess.get("outlines", [])
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
    ("story_bible", {"banned_phrases": [], "used_tropes": [], "open_threads": [], "established_facts": [], "paragraph_starters": [], "asked_questions": []}),
    # Widget defaults
    ("w_world_mode",      "作品世界"),
    ("w_world_input",     ""),
    ("w_character_notes", ""),
    ("w_environment",     ""),
    ("w_language",        "繁體中文"),
    ("w_perspective",     "第一人稱（我）"),
    ("w_length_label",    "中篇（約 2000 字）"),
    ("w_total_chapters",  5),
    ("num_cp_chars",      0),
    ("w_cp_type",         "無 CP"),
    ("w_love_tone",       "甜蜜溫馨"),
    ("w_pacing",          "張弛交替・積蓄爆發"),
    ("w_name",            ""),
    ("w_nickname",        ""),
    ("w_gender",          "女"),
    ("w_personality",     ""),
    ("w_appearance",      ""),
    ("w_residence",       ""),
    ("w_background",          ""),
    ("w_protagonist_facts",   ""),
    ("w_plot_want",           ""),
    ("w_plot_forbid",     ""),
    ("w_nsfw",            False),
    ("w_style_sample",    ""),
    ("w_style_reference", ""),
    ("w_next_next_dir",        ""),
    ("early_overview",         ""),
    ("early_overview_through", 0),
    ("_dir_ver", 0),
    ("outlines", []),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── API clients ───────────────────────────────────────────────────────────────

if not DEEPSEEK_API_KEY and not GEMINI_API_KEY:
    st.error("❌ 請在 .env 檔案中設定 DEEPSEEK_API_KEY 或 GEMINI_API_KEY。")
    st.stop()

_gemini_client = (
    OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    if GEMINI_API_KEY else None
)
_deepseek_client = (
    OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    if DEEPSEEK_API_KEY else None
)

def _fallback_client(primary):
    """Return the other client when primary hits rate limit."""
    if primary is _gemini_client and _deepseek_client:
        return _deepseek_client
    if primary is _deepseek_client and _gemini_client:
        return _gemini_client
    return primary

def _get_active_client() -> OpenAI:
    # NSFW mode: always use DeepSeek (Gemini has content filters)
    if st.session_state.get("w_nsfw", False) and _deepseek_client:
        return _deepseek_client
    sel = st.session_state.get("w_ai_provider", "Gemini")
    if sel == "Gemini" and _gemini_client:
        return _gemini_client
    if sel == "DeepSeek" and _deepseek_client:
        return _deepseek_client
    return _gemini_client or _deepseek_client

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📖 故事設定")

    # ── AI Provider ───────────────────────────────────────────────────────────
    st.markdown("### 🤖 AI 模型選擇")
    _provider_options = []
    if _gemini_client:
        _provider_options.append("Gemini")
    if _deepseek_client:
        _provider_options.append("DeepSeek")
    if not _provider_options:
        _provider_options = ["Gemini"]
    _default_provider = st.session_state.get("w_ai_provider", _provider_options[0])
    if _default_provider not in _provider_options:
        _default_provider = _provider_options[0]
    _selected = st.radio(
        "選擇 AI",
        _provider_options,
        index=_provider_options.index(_default_provider),
        horizontal=True,
        label_visibility="collapsed",
        key="w_ai_provider",
    )
    _nsfw_on = st.session_state.get("w_nsfw", False)
    if _nsfw_on and _deepseek_client:
        st.caption("🔞 限制級模式：自動使用 DeepSeek（Gemini 有內容過濾）")
    elif _selected == "Gemini":
        st.caption("✦ Gemini 2.0 Flash・免費・100 萬 token 記憶")
    else:
        st.caption("✦ DeepSeek Chat・付費・穩定備援")
    st.divider()

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
        cp_chars = []
        for i in range(st.session_state.num_cp_chars):
            cp_chars.append({
                "name":     st.session_state.get(f"cpname_{i}", ""),
                "intimacy": st.session_state.get(f"cpinti_{i}", "糖（甜蜜互動）"),
                "notes":    st.session_state.get(f"cpnotes_{i}", ""),
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
            "love_tone":        st.session_state.get("w_love_tone",         "甜蜜溫馨"),
            "pacing":           st.session_state.get("w_pacing",            "張弛交替・積蓄爆發"),
            "cp_characters":    cp_chars,
            "name":             st.session_state.get("w_name",              ""),
            "nickname":         st.session_state.get("w_nickname",          ""),
            "gender":           st.session_state.get("w_gender",            "女"),
            "personality":      st.session_state.get("w_personality",       ""),
            "appearance":       st.session_state.get("w_appearance",        ""),
            "residence":        st.session_state.get("w_residence",         ""),
            "background":       st.session_state.get("w_background",        ""),
            "protagonist_facts": st.session_state.get("w_protagonist_facts", ""),
            "plot_want":        st.session_state.get("w_plot_want",         ""),
            "plot_forbid":      st.session_state.get("w_plot_forbid",       ""),
            "nsfw":             st.session_state.get("w_nsfw",              False),
            "style_reference":  st.session_state.get("w_style_reference",   ""),
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
                        "perspective", "length_label", "total_chapters", "cp_type",
                        "love_tone", "pacing",
                        "name", "nickname", "gender", "personality", "appearance",
                        "residence", "background", "plot_want", "plot_forbid",
                        "environment", "nsfw", "style_reference"]:
                if key in data:
                    st.session_state[f"w_{key}"] = data[key]
            _cp_load = data.get("cp_characters", [])
            if not _cp_load and data.get("cp_character"):
                _cp_load = [{"name": data["cp_character"], "intimacy": data.get("intimacy", "糖（甜蜜互動）")}]
            st.session_state.num_cp_chars = len(_cp_load)
            for _j, _cp in enumerate(_cp_load):
                st.session_state[f"cpname_{_j}"] = _cp.get("name", "")
                st.session_state[f"cpinti_{_j}"] = _cp.get("intimacy", "糖（甜蜜互動）")
                st.session_state[f"cpnotes_{_j}"] = _cp.get("notes", "")
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

    # ── Novel file upload ─────────────────────────────────────────────────────
    st.markdown("### 📂 載入已下載文章")
    st.caption("上傳之前下載的 .txt，可繼續改寫或重新生成")
    _novel_file = st.file_uploader(
        "上傳 .txt 文章", type="txt", key="novel_file_upload",
        label_visibility="collapsed",
    )
    if _novel_file and _novel_file.name != st.session_state.get("_last_novel_upload"):
        _raw = _novel_file.read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        _parsed = [c.strip() for c in _raw.split("\n\n\n") if c.strip()]
        if _parsed:
            st.session_state.chapters    = _parsed
            st.session_state.summaries   = []
            st.session_state.story_bible = {"banned_phrases": [], "used_tropes": [], "open_threads": [], "established_facts": [], "paragraph_starters": [], "asked_questions": []}
            st.session_state.story_started = True
            st.session_state["_last_novel_upload"] = _novel_file.name
            try:
                with open(LAST_SESSION_FILE, "w", encoding="utf-8") as _sf:
                    json.dump({"chapters": _parsed, "summaries": [], "story_bible": st.session_state.story_bible, "saved_settings": st.session_state.saved_settings}, _sf, ensure_ascii=False, indent=2)
            except Exception:
                pass
            st.rerun()
        else:
            st.error("無法解析文章，請確認格式正確")

    st.divider()

    # ── Name replacer ─────────────────────────────────────────────────────────
    st.markdown("### 🔤 名字替換工具")
    st.caption("貼上文章或上傳 .txt，批量替換角色名字後下載")
    _rn_pasted = st.text_area(
        "貼上文章內容", key="rn_paste_area",
        placeholder="直接將文章內容貼到這裡…",
        height=160, label_visibility="collapsed",
    )
    st.caption("或上傳 .txt 檔案：")
    _rn_file = st.file_uploader(
        "上傳 .txt", type="txt", key="rename_file_upload",
        label_visibility="collapsed",
    )
    if _rn_file and _rn_file.name != st.session_state.get("_last_rn_upload"):
        st.session_state["_rn_text"] = _rn_file.read().decode("utf-8")
        st.session_state["_rn_filename"] = _rn_file.name
        st.session_state["_last_rn_upload"] = _rn_file.name
        st.session_state.pop("_rn_result", None)

    _rn_source = _rn_pasted.strip() or st.session_state.get("_rn_text", "")
    _rn_fname = st.session_state.get("_rn_filename", "renamed.txt") if not _rn_pasted.strip() else "renamed.txt"

    if _rn_source:
        st.caption(f"文章長度：{len(_rn_source)} 字")
        if "num_rn_pairs" not in st.session_state:
            st.session_state["num_rn_pairs"] = 2
        _rn_col1, _rn_col2 = st.columns(2)
        with _rn_col1:
            if st.button("＋ 新增替換", use_container_width=True):
                st.session_state["num_rn_pairs"] += 1
        with _rn_col2:
            if st.button("－ 移除替換", use_container_width=True,
                         disabled=st.session_state["num_rn_pairs"] <= 1):
                st.session_state["num_rn_pairs"] -= 1
        _rn_pairs = []
        for _ri in range(st.session_state["num_rn_pairs"]):
            _rc1, _rc2 = st.columns(2)
            with _rc1:
                _old = st.text_input("原名字", key=f"rn_old_{_ri}", placeholder="原名字")
            with _rc2:
                _new = st.text_input("新名字", key=f"rn_new_{_ri}", placeholder="新名字")
            if _old.strip() and _new.strip():
                _rn_pairs.append((_old.strip(), _new.strip()))
        if st.button("✨ 執行替換", type="primary", use_container_width=True,
                     disabled=not _rn_pairs):
            _rn_out = _rn_source
            for _old, _new in _rn_pairs:
                _rn_out = _rn_out.replace(_old, _new)
            st.session_state["_rn_result"] = _rn_out
            _replaced_count = sum(_rn_source.count(_old) for _old, _ in _rn_pairs)
            st.success(f"完成，共替換 {_replaced_count} 處")
        if st.session_state.get("_rn_result"):
            st.download_button(
                "⬇️ 下載替換後的文章",
                data=st.session_state["_rn_result"],
                file_name=_rn_fname,
                mime="text/plain",
                use_container_width=True,
            )

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
    pacing = st.selectbox(
        "敘事節奏",
        options=["緩節奏・情緒內斂", "緩節奏・情緒爆發", "快節奏・情緒內斂", "快節奏・情緒爆發", "張弛交替・積蓄爆發"],
        key="w_pacing",
        help="控制場景該快還是慢、情緒該藏還是爆",
    )

    st.divider()

    # ── CP ────────────────────────────────────────────────────────────────────
    st.markdown("### 💞 CP 設定")
    cp_type = st.radio(
        "配對", ["無 CP", "我 × 角色"],
        horizontal=True, label_visibility="collapsed", key="w_cp_type",
    )
    cp_characters = []
    if cp_type == "我 × 角色":
        col_cp_add, col_cp_del = st.columns(2)
        with col_cp_add:
            if st.button("＋ 新增 CP 對象", use_container_width=True):
                st.session_state.num_cp_chars += 1
        with col_cp_del:
            if st.button("－ 移除 CP 對象", use_container_width=True,
                         disabled=st.session_state.num_cp_chars == 0):
                st.session_state.num_cp_chars -= 1
        for i in range(st.session_state.num_cp_chars):
            with st.expander(f"CP 對象 {i + 1}", expanded=True):
                cp_name = st.text_input(
                    "角色名稱", key=f"cpname_{i}",
                    placeholder='輸入角色名，或輸入「隨機」由 AI 決定',
                )
                cp_inti = st.select_slider(
                    "親密程度",
                    options=["清水（純愛暗戀）", "糖（甜蜜互動）", "甜虐（曖昧張力）", "熾熱（激情親密）"],
                    key=f"cpinti_{i}",
                )
                cp_notes = st.text_area(
                    "感情備注（特殊限制）",
                    key=f"cpnotes_{i}",
                    placeholder="例：她是直女，即使到故事後期也不會輕易喜歡上主角，最多只是對主角有一絲困惑或好奇，不會主動表達好感…",
                    height=68,
                    help="填寫後，此限制凌駕通用感情節奏，AI 必須全程遵守。適合設定「直女慢慢動搖」「單方面曖昧」「極度緩慢的感情線」等情境。",
                )
                cp_characters.append({"name": cp_name, "intimacy": cp_inti, "notes": cp_notes})
        if not cp_characters:
            st.info("點擊「＋ 新增 CP 對象」加入配對角色")

    love_tone = st.selectbox(
        "戀愛情緒基調",
        options=["甜蜜溫馨", "歡喜冤家", "虐心糾纏", "青春悸動", "禁忌張力", "宿命糾纏"],
        key="w_love_tone",
        help="整個故事的戀愛情感氛圍走向",
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
    protagonist_facts = st.text_area(
        "主角固定事實",
        key="w_protagonist_facts",
        placeholder="每行一條，寫故事中已說過、不能被推翻的個人資訊。例如：\n媽媽對毛過敏，從沒養過寵物\n目前沒有喜歡的人\n從未去過日本\n不會游泳",
        height=100,
        help="這裡的每一條都會在每一章強制告訴 AI，絕對不能讓主角說出矛盾的話。",
    )

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

    # ── Style reference ───────────────────────────────────────────────────────
    st.markdown("### ✍️ 文風參考")
    st.caption("貼上你欣賞的作者文章片段，AI 將分析其文風並照此密度與選字寫作")

    _preset_options = ["（不使用預設）"] + list(STYLE_PRESETS.keys())
    _preset_sel = st.selectbox("套用預設文風", _preset_options, label_visibility="collapsed", key="_preset_sel")
    if _preset_sel != "（不使用預設）":
        if st.button("✦ 套用預設", use_container_width=True):
            st.session_state["w_style_reference"] = STYLE_PRESETS[_preset_sel]
            st.rerun()

    st.text_area(
        "參考文章片段", key="w_style_sample",
        placeholder="貼上任何你欣賞的中文小說段落（建議 200-1000 字）…",
        height=160, label_visibility="collapsed",
    )
    _sty_col1, _sty_col2 = st.columns(2)
    with _sty_col1:
        _analyze_btn = st.button("🔍 分析文風", use_container_width=True)
    with _sty_col2:
        if st.button("✕ 清除文風", use_container_width=True):
            st.session_state["w_style_reference"] = ""
            st.rerun()
    if _analyze_btn:
        _sample = st.session_state.get("w_style_sample", "").strip()
        if _sample:
            with st.spinner("分析文風中…"):
                _analysis = analyze_writing_style(client, _sample)
                _excerpt = _sample[:600].strip()
                st.session_state["w_style_reference"] = (
                    f"【參考原文節錄 — 直接照此句子長度、標點密度、動詞力度寫作】\n{_excerpt}\n\n"
                    f"【文風分析】\n{_analysis}"
                )
            st.rerun()
        else:
            st.warning("請先貼上參考文章片段")
    if st.session_state.get("w_style_reference"):
        st.success("✅ 文風已分析，生成時將套用")
        with st.expander("查看文風分析", expanded=False):
            st.write(st.session_state["w_style_reference"])

    st.divider()

    # ── Training notes ────────────────────────────────────────────────────────
    _all_notes = load_notes()
    st.markdown(f"### 🎓 訓練記錄（{len(_all_notes)} 條）")

    _tc1, _tc2 = st.columns(2)
    with _tc1:
        st.download_button(
            label="⬇️ 匯出訓練記錄",
            data=json.dumps(_all_notes, ensure_ascii=False, indent=2),
            file_name="training_notes.json",
            mime="application/json",
            use_container_width=True,
        )
    with _tc2:
        _tn_upload = st.file_uploader(
            "匯入訓練記錄", type="json", key="tn_upload",
            label_visibility="collapsed",
        )
        if _tn_upload is not None and _tn_upload.name != st.session_state.get("_last_tn_file"):
            try:
                _imported = json.loads(_tn_upload.read().decode("utf-8"))
                if isinstance(_imported, list):
                    from training import save_notes
                    save_notes(_imported)
                    st.session_state["_last_tn_file"] = _tn_upload.name
                    st.rerun()
                else:
                    st.error("格式錯誤")
            except Exception as _e:
                st.error(f"載入失敗：{_e}")

    if _all_notes:
        for _note in _all_notes:
            with st.expander(f"[{_note['type']}] {_note['issue'][:30]}…", expanded=False):
                if _note.get("excerpt"):
                    st.caption(f"問題段落：{_note['excerpt'][:120]}")
                st.write(_note["issue"])
                if st.button("🗑 刪除", key=f"del_{_note['id']}"):
                    delete_note(_note["id"])
                    st.rerun()
    else:
        st.caption("尚無訓練記錄。生成章節後，可在章節下方加入問題回饋。")

    st.divider()

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("### 📌 第一章大綱建議")
    st.caption("重置故事後，可在此填寫第一章的方向，生成大綱時自動帶入")
    st.text_area(
        "第一章大綱建議", key=f"w_first_dir_{st.session_state._dir_ver}",
        placeholder="例：開場在大雨夜、主角剛收到一封匿名信…（留空則由 AI 自由發揮）",
        height=80, label_visibility="collapsed",
    )
    start_btn = st.button("✨ 開始新故事大綱", type="primary", use_container_width=True)

# Auto-save sidebar settings on every run so refreshing restores current state
try:
    with open(LAST_SETTINGS_FILE, "w", encoding="utf-8") as _autosave_f:
        _autosave_f.write(_current_settings_json())
except Exception:
    pass

# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_settings() -> dict:
    return dict(
        world_mode=world_mode,
        world_input=world_input.strip(),
        language=language,
        perspective=perspective,
        length_label=length_label,
        cp_type=cp_type,
        pacing=pacing,
        love_tone=love_tone,
        cp_characters=cp_characters,
        name=name.strip(),
        nickname=nickname.strip(),
        gender=gender,
        personality=personality.strip(),
        appearance=appearance.strip(),
        background=background.strip(),
        extra_characters=extra_characters,
        protagonist_facts=protagonist_facts.strip(),
        plot_want=plot_want.strip(),
        plot_forbid=plot_forbid.strip(),
        total_chapters=int(total_chapters),
        character_notes=character_notes.strip(),
        environment=environment.strip(),
        residence=residence.strip(),
        nsfw=nsfw,
    )


def _save_session():
    try:
        data = {
            "chapters":              st.session_state.chapters,
            "summaries":             st.session_state.summaries,
            "story_bible":           st.session_state.story_bible,
            "saved_settings":        st.session_state.saved_settings,
            "early_overview":        st.session_state.get("early_overview", ""),
            "early_overview_through": st.session_state.get("early_overview_through", 0),
            "outlines":              st.session_state.get("outlines", []),
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
    if s["cp_type"] == "我 × 角色" and not any(c.get("name", "").strip() for c in s.get("cp_characters", [])):
        missing.append("至少一個配對角色名稱")
    return missing

# ── Character removal detection ──────────────────────────────────────────────
_cur_char_names: set[str] = set()
for _ec in extra_characters:
    if _ec.get("name", "").strip():
        _cur_char_names.add(_ec["name"].strip())
for _cp in cp_characters:
    _cpn = _cp.get("name", "").strip()
    if _cpn and _cpn != "隨機":
        _cur_char_names.add(_cpn)

_prev_char_names: set[str] = st.session_state.get("_known_char_names", set())
_removed_chars = _prev_char_names - _cur_char_names

if _removed_chars and st.session_state.get("chapters"):
    _sb = st.session_state.story_bible
    for _rn in _removed_chars:
        _sb["established_facts"] = [f for f in _sb.get("established_facts", []) if isinstance(f, str) and _rn not in f]
        _sb["open_threads"]      = [t for t in _sb.get("open_threads", [])      if isinstance(t, str) and _rn not in t]
        _sb["banned_phrases"]    = [p for p in _sb.get("banned_phrases", [])    if isinstance(p, str) and _rn not in p]
        _sb["used_tropes"]       = [t for t in _sb.get("used_tropes", [])       if isinstance(t, str) and _rn not in t]
    _sb.setdefault("removed_characters", [])
    for _rn in _removed_chars:
        if _rn not in _sb["removed_characters"]:
            _sb["removed_characters"].append(_rn)
    _save_session()

st.session_state["_known_char_names"] = _cur_char_names

# ── Generate logic ────────────────────────────────────────────────────────────

def _trim_to_original_end(original: str, processed: str, slack: int = 60) -> str:
    """Hard-trim processed text so it doesn't extend beyond original ending.
    Allows up to `slack` extra characters for legitimate in-place rewrites."""
    if len(processed) <= len(original) + slack:
        return processed
    target = len(original)
    break_pos = processed.rfind('\n\n', 0, target + slack)
    if break_pos != -1 and break_pos >= target - 300:
        return processed[:break_pos].rstrip()
    return processed[:target].rstrip()


_COMPRESS_THRESHOLD = 12
_RECENT_KEEP = 8


def _flatten_facts(raw):
    """Normalize established_facts from AI: handle dict/list/nested returns."""
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.extend(v for v in item.values() if isinstance(v, str))
        elif isinstance(item, list):
            out.extend(v for v in item if isinstance(v, str))
    return out


def _flatten_strs(raw):
    """Normalize a list field from AI: handle dict returns, keep only strings."""
    if isinstance(raw, dict):
        raw = list(raw.values())
    return [x for x in (raw if isinstance(raw, list) else []) if isinstance(x, str)]


def generate_chapter(settings: dict, chapter_num: int, prev_text: str = "", is_final: bool = False, directive: str = "", style_reference: str = "", preserve_ending: bool = False, outline: str = ""):
    full_text = ""
    placeholder = st.empty()
    active_client = _get_active_client()
    try:
        stream_iter = stream_chapter(
            client=active_client,
            chapter_num=chapter_num,
            prev_chapter_text=prev_text,
            is_final=is_final,
            prev_summaries=st.session_state.summaries,
            story_bible=st.session_state.story_bible,
            training_notes=load_notes(),
            chapter_directive=directive,
            style_reference=style_reference,
            early_overview=st.session_state.get("early_overview", ""),
            summary_offset=st.session_state.get("early_overview_through", 0),
            outline=outline,
            **settings,
        )
        for chunk in stream_iter:
            full_text += chunk
            placeholder.markdown(
                f'<div class="chapter-box">{full_text}</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        _err = str(e)
        _should_fallback = (
            "429" in _err
            or "402" in _err
            or "rate" in _err.lower()
            or "quota" in _err.lower()
            or "insufficient" in _err.lower()
            or "balance" in _err.lower()
        )
        if _should_fallback and _fallback_client(active_client) is not active_client:
            st.warning(f"⚠️ 主 API 無法使用（{_err[:80]}），自動切換備用 API 重試…")
            active_client = _fallback_client(active_client)
            full_text = ""
            for chunk in stream_chapter(
                client=active_client,
                chapter_num=chapter_num,
                prev_chapter_text=prev_text,
                is_final=is_final,
                prev_summaries=st.session_state.summaries,
                story_bible=st.session_state.story_bible,
                training_notes=load_notes(),
                chapter_directive=directive,
                style_reference=style_reference,
                early_overview=st.session_state.get("early_overview", ""),
                summary_offset=st.session_state.get("early_overview_through", 0),
                outline=outline,
                **settings,
            ):
                full_text += chunk
                placeholder.markdown(
                    f'<div class="chapter-box">{full_text}</div>',
                    unsafe_allow_html=True,
                )
        else:
            raise
    _pre_fix_len = len(full_text) if preserve_ending else 0
    if settings.get("nsfw"):
        with st.spinner("✍️ 修正重複段落…"):
            full_text = fix_repetitive_paragraphs(active_client, full_text, preserve_ending=preserve_ending)
            if preserve_ending:
                full_text = _trim_to_original_end(full_text[:_pre_fix_len], full_text)
            placeholder.markdown(
                f'<div class="chapter-box">{full_text}</div>',
                unsafe_allow_html=True,
            )
        with st.spinner("✍️ 改寫重複感知句式…"):
            full_text = fix_sensory_crutches(active_client, full_text, preserve_ending=preserve_ending)
            if preserve_ending:
                full_text = _trim_to_original_end(full_text[:_pre_fix_len], full_text)
            placeholder.markdown(
                f'<div class="chapter-box">{full_text}</div>',
                unsafe_allow_html=True,
            )
    if len(st.session_state.chapters) >= 1:
        with st.spinner("🔁 全文跨章重複段落偵測…"):
            full_text = fix_cross_chapter_repetition(active_client, full_text, st.session_state.chapters, nsfw=settings.get("nsfw", False))
            if preserve_ending:
                full_text = _trim_to_original_end(full_text[:_pre_fix_len], full_text)
            placeholder.markdown(
                f'<div class="chapter-box">{full_text}</div>',
                unsafe_allow_html=True,
            )
    with st.spinner("🔍 校正內容一致性…"):
        protagonist_name = settings.get("name", "") if settings else ""
        extra_names = [c["name"] for c in (settings.get("extra_characters") or []) if c.get("name", "").strip()]
        full_text = fix_consistency(active_client, full_text, protagonist_name, extra_names, settings.get("nickname", ""), preserve_ending=preserve_ending)
        if preserve_ending:
            full_text = _trim_to_original_end(full_text[:_pre_fix_len], full_text)
        placeholder.markdown(
            f'<div class="chapter-box">{full_text}</div>',
            unsafe_allow_html=True,
        )
    with st.spinner("📝 記錄章節記憶…"):
        summary, bible_update = summarize_chapter(active_client, full_text, chapter_num)
    st.session_state.summaries.append(summary)
    # Extract paragraph starters for cross-chapter dedup prevention
    b_pre = st.session_state.story_bible
    b_pre.setdefault("paragraph_starters", [])
    b_pre["paragraph_starters"].extend(extract_paragraph_starters(full_text))
    # Compress old summaries when list grows beyond threshold
    if len(st.session_state.summaries) > _COMPRESS_THRESHOLD:
        _old = st.session_state.summaries[:-_RECENT_KEEP]
        _existing_ov = st.session_state.get("early_overview", "")
        _ov_through = st.session_state.get("early_overview_through", 0)
        with st.spinner("📚 壓縮早期章節記憶…"):
            st.session_state.early_overview = compress_old_summaries(
                active_client, _existing_ov, _old, chapter_start=_ov_through + 1
            )
        st.session_state.early_overview_through = _ov_through + len(_old)
        st.session_state.summaries = st.session_state.summaries[-_RECENT_KEEP:]
    # Merge into story bible
    b = st.session_state.story_bible
    b["banned_phrases"] = (b["banned_phrases"] + bible_update.get("banned_phrases", []))[-40:]
    b["used_tropes"] = (b["used_tropes"] + bible_update.get("used_tropes", []))[-20:]
    b["open_threads"] = bible_update.get("open_threads", b["open_threads"])
    b.setdefault("established_facts", [])
    b["established_facts"] = (_flatten_facts(b["established_facts"]) + _flatten_facts(bible_update.get("established_facts", [])))[-80:]
    b.setdefault("asked_questions", [])
    b["asked_questions"] = list(dict.fromkeys(_flatten_strs(b["asked_questions"]) + _flatten_strs(bible_update.get("asked_questions", []))))
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
    _first_ch_hint = st.session_state.get(f"w_first_dir_{st.session_state._dir_ver}", "")
    st.session_state.chapters = []
    st.session_state.summaries = []
    st.session_state.outlines = []
    st.session_state.story_bible = {"banned_phrases": [], "used_tropes": [], "open_threads": [], "established_facts": [], "paragraph_starters": [], "asked_questions": []}
    st.session_state.early_overview = ""
    st.session_state.early_overview_through = 0
    st.session_state.saved_settings = s
    st.session_state.story_started = False
    # Pre-fill first chapter outline hint from sidebar input
    if _first_ch_hint.strip():
        st.session_state["ol_gen_0"] = _first_ch_hint
    # Clear per-chapter UI state and character tracking from previous story
    _stale_prefixes = ("editing_", "rewrite_inst_", "rewrite_as_ending_",
                       "t_excerpt_", "t_issue_", "t_correction_", "t_type_",
                       "inline_sum_", "_sum_edit_", "_save_sum_", "ol_edit_",
                       "ol_ver_", "ol_regen_done_", "ol_regen_ch_")
    for _k in [k for k in st.session_state if any(k.startswith(p) for p in _stale_prefixes)]:
        del st.session_state[_k]
    st.session_state["_known_char_names"] = set()
    st.session_state._dir_ver += 1
    try:
        with open(LAST_SESSION_FILE, "w", encoding="utf-8") as _f:
            import json as _json2
            _json2.dump({"chapters": [], "summaries": [], "outlines": [],
                         "story_bible": st.session_state.story_bible,
                         "saved_settings": {}, "early_overview": "", "early_overview_through": 0}, _f,
                        ensure_ascii=False)
    except Exception:
        pass
    st.success("✅ 故事已重置！請到「📋 大綱規劃」分頁開始生成大綱。")
    st.rerun()

# ── Main display ──────────────────────────────────────────────────────────────

st.markdown("## 📖 小說連載生成器")
st.caption("作品同人 × 原創世界 × 連載章節，把自己寫進故事裡")

s = st.session_state.saved_settings
_has_settings = bool(s.get("world_input", ""))
_live_settings_ready = bool(world_input.strip()) and bool(name.strip())

if _has_settings:
    world_label = f"《{s['world_input']}》" if s["world_mode"] == "作品世界" else s["world_input"][:30] + "…"
    _cp_names = "、".join(c["name"] for c in s.get("cp_characters", []) if c.get("name", "").strip())
    cp_label = f"♡ {_cp_names}" if s["cp_type"] == "我 × 角色" and _cp_names else "無 CP"
    st.markdown(f"**{world_label}**　·　{s['name']}　·　{s['language']}　·　{s['length_label']}　·　{cp_label}")
elif st.session_state.chapters:
    st.info("已載入文章。若需要 AI 重新生成，請一併在左側載入設定檔（.json）。")

_outline_count = len(st.session_state.get("outlines", []))
if st.session_state.chapters or _outline_count or _has_settings:
    st.caption(f"共 {len(st.session_state.chapters)} 章" + (f"　|　大綱 {_outline_count} 個" if _outline_count else ""))
    st.divider()

_tab_outlines, _tab_chapters, _tab_summaries = st.tabs(["📋 大綱規劃", "📖 章節", "📝 摘要管理"])

with _tab_outlines:
    st.markdown("#### 📋 大綱規劃")
    st.caption("預先生成各章節大綱，生成章節時自動依此撰寫。可隨時修改或重新生成各章大綱。")

    _ol_c1, _ol_c2 = st.columns([3, 1])
    with _ol_c1:
        _ol_start_num = len(st.session_state.get("outlines", [])) + 1
        _ol_count_input = st.number_input(
            f"生成章數（從第 {_ol_start_num} 章開始）",
            min_value=1, max_value=50, value=10, step=1,
            key="_outline_gen_count",
        )
    with _ol_c2:
        st.write("")
        _ol_gen_btn = st.button(
            "✨ 生成大綱", type="primary", use_container_width=True,
            key="_outline_gen_btn",
            disabled=not _live_settings_ready,
        )
    st.text_area(
        "整批生成建議（選填）",
        key="_ol_batch_hint",
        placeholder=(
            "例：這幾章圍繞一個核心誤會推進，感情線從疏遠到靠近\n"
            "例：每章結尾都留下一個未解決的懸念，帶入下一章\n"
            "例：主線是調查某件事，每章揭露一部分真相"
        ),
        height=80,
        help="填寫後，這批大綱都會按此方向生成，並環環相扣",
    )
    if not _live_settings_ready:
        st.caption("請先在左側填寫作品名稱與主角姓名，才能生成大綱。")

    if _ol_gen_btn:
        _s_ol = _collect_settings()
        st.session_state.saved_settings = _s_ol
        _start_idx = len(st.session_state.outlines)
        _end_idx = _start_idx + int(_ol_count_input)
        _prog = st.progress(0, text=f"生成大綱中…（0/{int(_ol_count_input)}）")
        for _i_ol, _ch_num_ol in enumerate(range(_start_idx + 1, _end_idx + 1)):
            _ol_text = ""
            _ol_ph = st.empty()
            _prev_ols_str = "\n\n".join(
                f"【第 {_j+1} 章大綱】\n{_ol}"
                for _j, _ol in enumerate(st.session_state.outlines)
            )
            try:
                _bulk_gen_hint = st.session_state.get(f"ol_gen_{_ch_num_ol - 1}", "")
                for _chunk in stream_outline(
                    client=_get_active_client(),
                    chapter_num=_ch_num_ol,
                    prev_summaries=st.session_state.summaries,
                    story_bible=st.session_state.story_bible,
                    early_overview=st.session_state.get("early_overview", ""),
                    summary_offset=st.session_state.get("early_overview_through", 0),
                    prev_outlines=_prev_ols_str,
                    is_final=(_ch_num_ol >= _s_ol["total_chapters"]),
                    chapter_directive=_bulk_gen_hint,
                    batch_hint=st.session_state.get("_ol_batch_hint", ""),
                    **_s_ol,
                ):
                    _ol_text += _chunk
                    _ol_ph.markdown(f"**第 {_ch_num_ol} 章大綱** ✍️\n\n{_ol_text}")
                while len(st.session_state.outlines) < _ch_num_ol:
                    st.session_state.outlines.append("")
                st.session_state.outlines[_ch_num_ol - 1] = _ol_text
                _ol_ph.empty()
            except Exception as _oe:
                _ol_ph.error(f"第 {_ch_num_ol} 章大綱生成失敗：{_oe}")
                break
            _prog.progress((_i_ol + 1) / int(_ol_count_input),
                           text=f"生成大綱中…（{_i_ol + 1}/{int(_ol_count_input)}）")
        _prog.empty()
        _save_session()
        st.rerun()

    if st.session_state.get("outlines"):
        st.divider()
        _pending_ol_delete = None
        _pending_ol_insert = None
        for _oi, _oval in enumerate(st.session_state.outlines):
            _ch_done = _oi < len(st.session_state.chapters)
            _badge = " ✅" if _ch_done else ""
            _just_regen = st.session_state.pop(f"ol_regen_done_{_oi}", False)
            if _just_regen:
                _ver = st.session_state.get(f"ol_ver_{_oi}", 0) + 1
                st.session_state[f"ol_ver_{_oi}"] = _ver
            else:
                _ver = st.session_state.get(f"ol_ver_{_oi}", 0)
            with st.expander(f"第 {_oi + 1} 章大綱{_badge}", expanded=_just_regen or not _ch_done):
                _ol_edited = st.text_area(
                    "大綱",
                    value=_oval,
                    key=f"ol_edit_{_oi}_v{_ver}",
                    height=200,
                    label_visibility="collapsed",
                )
                _hint_c1, _hint_c2 = st.columns(2)
                with _hint_c1:
                    st.caption("✏️ 生成建議（希望這章包含的方向或元素）")
                    _ol_gen = st.text_area(
                        "生成建議",
                        key=f"ol_gen_{_oi}",
                        placeholder="例：這章要有兩人在雨中相遇\n例：結尾安排誤會事件\n例：出現第三個角色破壞氣氛",
                        height=90,
                        label_visibility="collapsed",
                    )
                with _hint_c2:
                    st.caption("🔧 修正建議（大綱中的設定錯誤，重生成時強制修正）")
                    _ol_note = st.text_area(
                        "修正建議",
                        key=f"ol_note_{_oi}",
                        placeholder="例：Leo不是員工，他跟所有人都不認識\n例：主角不會開車\n例：這章不要出現室內場景",
                        height=90,
                        label_visibility="collapsed",
                    )
                _rg_ph = st.empty()
                _ol_sv, _ol_rg, _ol_ins, _ol_dl = st.columns(4)
                with _ol_sv:
                    if st.button("💾 儲存", key=f"ol_save_{_oi}_v{_ver}", use_container_width=True):
                        st.session_state.outlines[_oi] = _ol_edited
                        _save_session()
                        st.success("已儲存")
                with _ol_rg:
                    if st.button("🔄 重生成", key=f"ol_regen_{_oi}", use_container_width=True,
                                 disabled=not _live_settings_ready):
                        _s_rg = _collect_settings()
                        _prev_ols_rg = "\n\n".join(
                            f"【第 {_j+1} 章大綱】\n{_ol}"
                            for _j, _ol in enumerate(st.session_state.outlines) if _j != _oi
                        )
                        _new_ol = ""
                        try:
                            for _chunk in stream_outline(
                                client=_get_active_client(),
                                chapter_num=_oi + 1,
                                prev_summaries=st.session_state.summaries,
                                story_bible=st.session_state.story_bible,
                                early_overview=st.session_state.get("early_overview", ""),
                                summary_offset=st.session_state.get("early_overview_through", 0),
                                prev_outlines=_prev_ols_rg,
                                is_final=(_oi + 1 >= _s_rg["total_chapters"]),
                                chapter_directive=_ol_gen,
                                outline_note=_ol_note,
                                **_s_rg,
                            ):
                                _new_ol += _chunk
                                _rg_ph.markdown(f"**重生成第 {_oi + 1} 章大綱**\n\n{_new_ol}")
                            if _new_ol:
                                _outlines = list(st.session_state.outlines)
                                _outlines[_oi] = _new_ol
                                st.session_state.outlines = _outlines
                                st.session_state[f"ol_regen_done_{_oi}"] = True
                                _save_session()
                                st.rerun()
                            else:
                                st.warning("生成結果為空，請再試一次。")
                        except Exception as _oe:
                            st.error(f"生成失敗：{_oe}")
                with _ol_ins:
                    if st.button("➕ 插入", key=f"ol_insert_{_oi}", use_container_width=True,
                                 help="在此章前插入一個空白大綱"):
                        _pending_ol_insert = _oi
                with _ol_dl:
                    if st.button("🗑 刪除", key=f"ol_del_{_oi}", use_container_width=True):
                        _pending_ol_delete = _oi
                _can_gen_ch = (
                    not _ch_done
                    and _live_settings_ready
                )
                if _ch_done:
                    st.caption("✅ 本章已生成，至「章節」分頁查看")
                    if st.button(
                        "🔄 依更新大綱重新生成",
                        key=f"ol_regen_ch_{_oi}",
                        use_container_width=True,
                        disabled=not _live_settings_ready,
                        help="修改大綱後點此依新大綱重新生成本章（會覆蓋現有章節）",
                    ):
                        _s_regen_ch = _collect_settings()
                        missing_regen = _validate(_s_regen_ch)
                        if missing_regen:
                            st.warning(f"請填寫：{'、'.join(missing_regen)}")
                        else:
                            st.session_state.saved_settings = _s_regen_ch
                            _ch_num_regen = _oi + 1
                            _is_final_regen = _ch_num_regen >= _s_regen_ch["total_chapters"]
                            _prev_text_regen = st.session_state.chapters[_oi - 1] if _oi > 0 else ""
                            _outline_val_regen = st.session_state.get(f"ol_edit_{_oi}_v{_ver}", _oval)
                            _ov_through_regen = st.session_state.get("early_overview_through", 0)
                            if _ov_through_regen >= _oi + 1:
                                st.session_state.early_overview = ""
                                st.session_state.early_overview_through = 0
                                st.session_state.summaries = []
                            else:
                                _keep_regen = _oi - _ov_through_regen
                                st.session_state.summaries = st.session_state.summaries[:_keep_regen]
                            chapter_text = generate_chapter(
                                _s_regen_ch,
                                chapter_num=_ch_num_regen,
                                prev_text=_prev_text_regen,
                                is_final=_is_final_regen,
                                style_reference=st.session_state.get("w_style_reference", ""),
                                outline=_outline_val_regen,
                            )
                            st.session_state.chapters[_oi] = chapter_text
                            _save_session()
                            st.rerun()
                if st.button(
                    "✨ 依此大綱生成章節",
                    key=f"ol_gen_ch_{_oi}",
                    use_container_width=True,
                    disabled=not _can_gen_ch,
                    type="primary" if _can_gen_ch else "secondary",
                ):
                    _s_gch = _collect_settings()
                    missing_gch = _validate(_s_gch)
                    if missing_gch:
                        st.warning(f"請填寫：{'、'.join(missing_gch)}")
                    else:
                        st.session_state.saved_settings = _s_gch
                        _ch_num_gen = _oi + 1
                        _is_final_gen = _ch_num_gen >= _s_gch["total_chapters"]
                        _prev_text_gen = (
                            st.session_state.chapters[_oi - 1]
                            if _oi > 0 and len(st.session_state.chapters) > _oi - 1
                               and st.session_state.chapters[_oi - 1]
                            else ""
                        )
                        _outline_val = st.session_state.get(f"ol_edit_{_oi}_v{_ver}", _oval)
                        st.session_state.story_started = True
                        chapter_text = generate_chapter(
                            _s_gch,
                            chapter_num=_ch_num_gen,
                            prev_text=_prev_text_gen,
                            is_final=_is_final_gen,
                            style_reference=st.session_state.get("w_style_reference", ""),
                            outline=_outline_val,
                        )
                        # Pad with empty strings if generating out of order
                        while len(st.session_state.chapters) < _oi:
                            st.session_state.chapters.append("")
                        if len(st.session_state.chapters) == _oi:
                            st.session_state.chapters.append(chapter_text)
                        else:
                            st.session_state.chapters[_oi] = chapter_text
                        _save_session()
                        st.rerun()
        if _pending_ol_insert is not None:
            st.session_state.outlines.insert(_pending_ol_insert, "")
            _save_session()
            st.rerun()
        elif _pending_ol_delete is not None:
            st.session_state.outlines.pop(_pending_ol_delete)
            _save_session()
            st.rerun()
        st.divider()
        if st.button("🗑 清除所有大綱", key="_ol_clear_all"):
            st.session_state.outlines = []
            _save_session()
            st.rerun()
    else:
        st.info("尚無大綱。點擊「生成大綱」規劃各章節大綱，生成章節時將自動依此撰寫。")

with _tab_summaries:
    st.markdown("#### 📝 章節摘要管理")
    if not st.session_state.chapters:
        st.caption("尚無章節。生成章節後自動記錄摘要。")
    else:
        st.caption("AI 自動生成的摘要，供後續章節生成時參考。可直接修改，避免 AI 誤記。")
        _ov_through = st.session_state.get("early_overview_through", 0)
        if st.session_state.get("early_overview"):
            st.markdown(f"**早期故事總覽（第 1～{_ov_through} 章）**")
            _ov_val = st.text_area(
                "早期故事總覽",
                value=st.session_state.early_overview,
                key="_edit_early_ov",
                height=200,
                label_visibility="collapsed",
            )
            if st.button("💾 儲存早期總覽", key="_save_early_ov"):
                st.session_state.early_overview = _ov_val
                _save_session()
                st.success("已儲存")
            st.divider()
        if st.session_state.summaries:
            for _si, _sval in enumerate(st.session_state.summaries):
                _ch_label = f"第 {_ov_through + _si + 1} 章摘要"
                with st.expander(_ch_label, expanded=False):
                    _edited_s = st.text_area(
                        "摘要內容",
                        value=_sval,
                        key=f"_sum_edit_{_si}",
                        height=160,
                        label_visibility="collapsed",
                    )
                    if st.button("💾 儲存", key=f"_save_sum_{_si}"):
                        st.session_state.summaries[_si] = _edited_s
                        _save_session()
                        st.success("已儲存")
        elif not st.session_state.get("early_overview"):
            st.caption("尚無摘要。生成章節後自動記錄。")

        st.divider()
        st.markdown("#### 📚 故事資料庫管理")
        st.caption("若 AI 記錯了事實（如角色位置、能力狀態），可在此直接修改或刪除，下次生成時立即生效。刪除已移除角色的相關條目，可防止 AI 繼續引用。")
        _b = st.session_state.story_bible
        with st.expander("已確立的故事事實 (established_facts)", expanded=False):
            _ef_text = "\n".join(x for x in _b.get("established_facts", []) if isinstance(x, str))
            _ef_edited = st.text_area(
                "每行一條事實",
                value=_ef_text,
                key="_edit_ef",
                height=200,
                label_visibility="collapsed",
                placeholder="每行一條，例如：\n主角目前在機場\nDalon目前在辦公室\n主角不知道Dalon的秘密",
            )
            if st.button("💾 儲存事實", key="_save_ef"):
                _b["established_facts"] = [l.strip() for l in _ef_edited.split("\n") if l.strip()]
                _save_session()
                st.success("已儲存")
        with st.expander("未解決的伏筆 (open_threads)", expanded=False):
            _ot_text = "\n".join(_b.get("open_threads", []))
            _ot_edited = st.text_area(
                "每行一條伏筆",
                value=_ot_text,
                key="_edit_ot",
                height=120,
                label_visibility="collapsed",
            )
            if st.button("💾 儲存伏筆", key="_save_ot"):
                _b["open_threads"] = [l.strip() for l in _ot_edited.split("\n") if l.strip()]
                _save_session()
                st.success("已儲存")

with _tab_chapters:
    if not st.session_state.chapters and not st.session_state.story_started:
        st.markdown("""
<div style="text-align:center;padding:80px 20px;color:#475569">
  <div style="font-size:3.5rem">📖</div>
  <div style="font-size:1.15rem;margin-top:14px;font-weight:600">在左側填寫設定，開始你的故事</div>
  <div style="font-size:.92rem;margin-top:8px;color:#64748b">
    支援任何作品世界 · 原創設定 · 連載章節<br>可先到「大綱規劃」分頁預先規劃章節大綱
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        for i, text in enumerate(st.session_state.chapters):
            if not text:
                continue
            st.markdown(f'<div id="ch-{i}"></div>', unsafe_allow_html=True)
            st.markdown(f"## 第 {i + 1} 章")
            st.markdown(f'<div class="chapter-box">{text}</div>', unsafe_allow_html=True)

            _regen_col, _edit_col, _sum_col = st.columns(3)
            with _regen_col:
                _regen_btn = st.button(
                    "🔄 重新生成本章", key=f"regen_{i}", use_container_width=True,
                    disabled=not _has_settings,
                )
            with _edit_col:
                _edit_btn = st.button("✏️ 改寫本章", key=f"edit_toggle_{i}", use_container_width=True)
            with _sum_col:
                _sum_btn = st.button("📝 製作摘要", key=f"sum_btn_{i}", use_container_width=True)

            if _edit_btn:
                _cur = st.session_state.get(f"editing_{i}", False)
                st.session_state[f"editing_{i}"] = not _cur
                st.rerun()

            if _sum_btn:
                with st.spinner(f"📝 為第 {i+1} 章製作摘要…"):
                    _new_sum, _new_bible = summarize_chapter(client, text, i + 1)
                _ov_t = st.session_state.get("early_overview_through", 0)
                _sidx = i - _ov_t
                if _sidx >= 0:
                    while len(st.session_state.summaries) <= _sidx:
                        st.session_state.summaries.append("")
                    st.session_state.summaries[_sidx] = _new_sum
                    b = st.session_state.story_bible
                    b["banned_phrases"] = (b["banned_phrases"] + _new_bible.get("banned_phrases", []))[-40:]
                    b["used_tropes"] = (b["used_tropes"] + _new_bible.get("used_tropes", []))[-20:]
                    b.setdefault("established_facts", [])
                    b["established_facts"] = (_flatten_facts(b["established_facts"]) + _flatten_facts(_new_bible.get("established_facts", [])))[-80:]
                    b.setdefault("asked_questions", [])
                    b["asked_questions"] = list(dict.fromkeys(_flatten_strs(b["asked_questions"]) + _flatten_strs(_new_bible.get("asked_questions", []))))
                    if _new_bible.get("open_threads"):
                        b["open_threads"] = _new_bible["open_threads"]
                    _save_session()
                    st.rerun()
                else:
                    st.warning("此章節已被壓縮進早期總覽，請至「摘要管理」分頁直接編輯早期總覽。")

            # Show inline summary if it exists for this chapter
            _ov_t2 = st.session_state.get("early_overview_through", 0)
            _sidx2 = i - _ov_t2
            if 0 <= _sidx2 < len(st.session_state.summaries) and st.session_state.summaries[_sidx2]:
                with st.expander("📋 本章摘要", expanded=False):
                    _sum_display = st.text_area(
                        "摘要（可直接修改）",
                        value=st.session_state.summaries[_sidx2],
                        key=f"inline_sum_{i}",
                        height=120,
                        label_visibility="collapsed",
                    )
                    if st.button("💾 儲存修改", key=f"inline_sum_save_{i}"):
                        st.session_state.summaries[_sidx2] = _sum_display
                        _save_session()
                        st.success("已儲存")

            if st.session_state.get(f"editing_{i}", False):
                with st.container():
                    _edited_text = st.text_area(
                        "編輯章節內容",
                        value=text,
                        key=f"edit_area_{i}",
                        height=500,
                        label_visibility="collapsed",
                    )
                    _save_col, _cancel_col = st.columns(2)
                    with _save_col:
                        if st.button("💾 儲存改寫", key=f"save_edit_{i}", type="primary", use_container_width=True):
                            st.session_state.chapters[i] = _edited_text
                            st.session_state[f"editing_{i}"] = False
                            _save_session()
                            st.rerun()
                    with _cancel_col:
                        if st.button("✕ 取消", key=f"cancel_edit_{i}", use_container_width=True):
                            st.session_state[f"editing_{i}"] = False
                            st.rerun()

            if _regen_btn:
                s = _collect_settings()
                st.session_state.saved_settings = s
                prev_text = st.session_state.chapters[i - 1] if i > 0 else ""
                is_last = i == len(st.session_state.chapters) - 1
                is_final = is_last and not st.session_state.story_started
                _ov_through = st.session_state.get("early_overview_through", 0)
                if _ov_through >= i + 1:
                    st.session_state.early_overview = ""
                    st.session_state.early_overview_through = 0
                    st.session_state.summaries = []
                else:
                    _keep = i - _ov_through
                    st.session_state.summaries = st.session_state.summaries[:_keep]
                _regen_inst = st.session_state.get(f"rewrite_inst_{i}", "").strip()
                if _regen_inst and st.session_state.get(f"rewrite_as_ending_{i}", False):
                    _regen_inst += _DIRECTIVE_AS_ENDING_RULE
                new_text = generate_chapter(s, chapter_num=i + 1, prev_text=prev_text, is_final=is_final,
                                            directive=_regen_inst,
                                            style_reference=st.session_state.get("w_style_reference", ""))
                st.session_state.chapters[i] = new_text
                st.rerun()

            # ── AI rewrite ────────────────────────────────────────────────────
            with st.expander("✍️ AI 改寫本章"):
                _inst_key = f"rewrite_inst_{i}"
                st.text_area(
                    "改寫指示",
                    key=_inst_key,
                    placeholder="例：把這章結局改成兩人爭吵離場、讓節奏更緊湊、刪掉中間的閒聊加強衝突…",
                    height=80,
                    label_visibility="collapsed",
                )
                st.checkbox(
                    "以此指示作為本章結尾（AI 寫到指示事件後立即收章）",
                    key=f"rewrite_as_ending_{i}",
                )
                if st.button("✨ 依指示重新生成", key=f"rewrite_btn_{i}",
                             use_container_width=True, disabled=not _has_settings):
                    _inst = st.session_state.get(_inst_key, "").strip()
                    if _inst:
                        _rewrite_as_ending = st.session_state.get(f"rewrite_as_ending_{i}", False)
                        if _rewrite_as_ending:
                            _inst += _DIRECTIVE_AS_ENDING_RULE
                        _s = _collect_settings()
                        st.session_state.saved_settings = _s
                        _prev = st.session_state.chapters[i - 1] if i > 0 else ""
                        _is_last = i == len(st.session_state.chapters) - 1
                        _is_final = _is_last and not st.session_state.story_started
                        _ov_through_rw = st.session_state.get("early_overview_through", 0)
                        if _ov_through_rw >= i + 1:
                            st.session_state.early_overview = ""
                            st.session_state.early_overview_through = 0
                            st.session_state.summaries = []
                        else:
                            _keep_rw = i - _ov_through_rw
                            st.session_state.summaries = st.session_state.summaries[:_keep_rw]
                        _new_text = generate_chapter(
                            _s, chapter_num=i + 1, prev_text=_prev, is_final=_is_final,
                            directive=_inst,
                            style_reference=st.session_state.get("w_style_reference", ""),
                            preserve_ending=_rewrite_as_ending,
                        )
                        st.session_state.chapters[i] = _new_text
                        st.rerun()
                    else:
                        st.warning("請填寫改寫指示")

            with st.expander("🎓 訓練回饋：標記本章問題"):
                _t_excerpt = st.text_area(
                    "貼上有問題的段落（可留空）",
                    key=f"t_excerpt_{i}", height=80,
                    placeholder="從章節中複製有問題的句子或段落，貼到這裡…",
                )
                _t_issue = st.text_area(
                    "說明問題（發生了什麼錯誤）",
                    key=f"t_issue_{i}", height=60,
                    placeholder="例：角色明明在室外，下一句卻在房間裡說話…",
                )
                _t_correction = st.text_area(
                    "正確做法應該是…（填了會讓 AI 更容易遵守）",
                    key=f"t_correction_{i}", height=60,
                    placeholder="例：角色離開室外後，必須先交代進入室內的過程，才能讓她在室內開口說話…",
                )
                _t_type = st.selectbox("問題類型", ISSUE_TYPES, key=f"t_type_{i}")
                if st.button("✅ 加入訓練記錄", key=f"t_submit_{i}"):
                    if _t_issue.strip():
                        add_note(_t_excerpt, _t_issue, _t_type, _t_correction)
                        st.success("已記錄！下次生成時 AI 將遵守此規則。")
                        st.rerun()
                    else:
                        st.warning("請填寫問題說明。")

        # ── Continue / Ending buttons (inside tab so generation streams here) ──
        if st.session_state.story_started:
            st.divider()
            if not _has_settings:
                st.warning("載入設定檔（.json）後，才能使用 AI 繼續生成下一章或結局。")
            st.markdown("#### 📌 下一章特別指示")
            st.text_area(
                "下一章特別指示", key=f"next_dir_{st.session_state._dir_ver}",
                placeholder="例：這章以回憶展開、發生停電意外、CP 獨處…（留空則由 AI 自由發揮）",
                height=80, label_visibility="collapsed",
            )
            st.checkbox(
                "以此指示作為本章結尾（AI 寫到指示事件後立即收章）",
                key=f"next_dir_as_ending_{st.session_state._dir_ver}",
            )
            st.markdown("#### 📌 下下章特別指示")
            st.caption("生成下一章後自動移入上方「下一章特別指示」")
            st.text_area(
                "下下章特別指示", key="w_next_next_dir",
                placeholder="預先填入，生成下一章後自動帶入…",
                height=80, label_visibility="collapsed",
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                continue_btn = st.button("➡️ 繼續下一章", type="primary", use_container_width=True, disabled=not _has_settings)
            with btn_col2:
                ending_btn = st.button("🎬 寫結局", use_container_width=True, disabled=not _has_settings)

            if continue_btn:
                s = _collect_settings()
                st.session_state.saved_settings = s
                chapter_num = len(st.session_state.chapters) + 1
                is_final = chapter_num >= s["total_chapters"]
                _directive = st.session_state.get(f"next_dir_{st.session_state._dir_ver}", "")
                _dir_as_ending = st.session_state.get(f"next_dir_as_ending_{st.session_state._dir_ver}", False)
                _next_next = st.session_state.get("w_next_next_dir", "")
                if _dir_as_ending and _directive.strip():
                    _directive += _DIRECTIVE_AS_ENDING_RULE
                st.session_state._dir_ver += 1
                if _next_next.strip():
                    st.session_state[f"next_dir_{st.session_state._dir_ver}"] = _next_next
                    st.session_state["w_next_next_dir"] = ""
                _outlines = st.session_state.get("outlines", [])
                _outline = _outlines[chapter_num - 1] if len(_outlines) >= chapter_num else ""
                chapter_text = generate_chapter(s, chapter_num=chapter_num,
                                                prev_text=st.session_state.chapters[-1],
                                                is_final=is_final, directive=_directive,
                                                style_reference=st.session_state.get("w_style_reference", ""),
                                                preserve_ending=_dir_as_ending, outline=_outline)
                st.session_state.chapters.append(chapter_text)
                _save_session()
                st.rerun()

            if ending_btn:
                s = _collect_settings()
                st.session_state.saved_settings = s
                chapter_num = len(st.session_state.chapters) + 1
                _directive = st.session_state.get(f"next_dir_{st.session_state._dir_ver}", "")
                _dir_as_ending = st.session_state.get(f"next_dir_as_ending_{st.session_state._dir_ver}", False)
                if _dir_as_ending and _directive.strip():
                    _directive += _DIRECTIVE_AS_ENDING_RULE
                st.session_state._dir_ver += 1
                st.session_state["w_next_next_dir"] = ""
                chapter_text = generate_chapter(s, chapter_num=chapter_num,
                                                prev_text=st.session_state.chapters[-1],
                                                is_final=True, directive=_directive,
                                                style_reference=st.session_state.get("w_style_reference", ""),
                                                preserve_ending=_dir_as_ending)
                st.session_state.chapters.append(chapter_text)
                _save_session()
                st.session_state.story_started = False
                st.rerun()

    # ── Chapter navigator (floating right panel) ──────────────────────────────
    _nav_titles = []
    for _ci, _ct in enumerate(st.session_state.chapters):
        _first = _ct.strip().split('\n')[0].strip()
        _nav_titles.append(_first[:22] if _first else f"第 {_ci+1} 章")
    _nav_json = json.dumps(_nav_titles, ensure_ascii=False)
    components.html(f"""
<script>
(function(){{
  var chapters = {_nav_json};
  var D = window.parent.document;

  var old = D.getElementById('_novel_nav');
  var wasCollapsed = old ? old.dataset.collapsed === '1' : false;
  if (old) old.remove();

  var nav = D.createElement('div');
  nav.id = '_novel_nav';
  nav.dataset.collapsed = wasCollapsed ? '1' : '0';
  nav.style.cssText = [
    'position:fixed','right:20px','top:72px','z-index:9999',
    'background:rgba(15,15,15,0.93)','border:1px solid rgba(255,255,255,0.1)',
    'border-radius:14px','padding:8px 6px',
    'backdrop-filter:blur(10px)','box-shadow:0 4px 24px rgba(0,0,0,0.4)',
    'transition:all 0.2s ease'
  ].join(';');

  /* ── header row (always visible) ── */
  var hd = D.createElement('div');
  hd.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:0 4px 6px;gap:8px;white-space:nowrap';

  var label = D.createElement('span');
  label.style.cssText = 'font-size:0.7rem;color:#555;font-weight:700;letter-spacing:1px';
  label.textContent = '📖  章節';

  var arrow = D.createElement('button');
  arrow.id = '_novel_nav_arrow';
  arrow.style.cssText = [
    'background:none','border:none','color:#666','cursor:pointer',
    'font-size:0.75rem','padding:0 2px','line-height:1','transition:transform 0.2s'
  ].join(';');

  hd.appendChild(label);
  hd.appendChild(arrow);
  nav.appendChild(hd);

  /* ── chapter list ── */
  var list = D.createElement('div');
  list.id = '_novel_nav_list';

  chapters.forEach(function(title, i){{
    var btn = D.createElement('button');
    btn.textContent = title;
    btn.style.cssText = [
      'display:block','width:100%','background:none','border:none',
      'color:#aaa','padding:5px 8px','text-align:left','cursor:pointer',
      'font-size:0.78rem','border-radius:7px','margin-bottom:2px',
      'white-space:nowrap','overflow:hidden','text-overflow:ellipsis','max-width:160px'
    ].join(';');
    btn.onmouseover = function(){{ this.style.background='rgba(255,255,255,0.09)'; this.style.color='#fff'; }};
    btn.onmouseout  = function(){{ this.style.background='none'; this.style.color='#aaa'; }};
    btn.onclick = function(){{
      var el = D.getElementById('ch-'+i);
      if (el) el.scrollIntoView({{behavior:'smooth', block:'start'}});
    }};
    list.appendChild(btn);
  }});
  nav.appendChild(list);

  /* ── collapse / expand logic ── */
  function applyState(){{
    var collapsed = nav.dataset.collapsed === '1';
    list.style.display = collapsed ? 'none' : 'block';
    label.style.display = collapsed ? 'none' : 'inline';
    arrow.textContent   = collapsed ? '◀' : '▶';
    nav.style.minWidth  = collapsed ? '0' : '124px';
    nav.style.maxHeight = collapsed ? 'none' : '72vh';
    nav.style.overflowY = collapsed ? 'visible' : 'auto';
    nav.style.padding   = collapsed ? '8px 6px' : '8px 6px';
  }}

  arrow.onclick = function(){{
    nav.dataset.collapsed = nav.dataset.collapsed === '1' ? '0' : '1';
    applyState();
  }};

  applyState();
  D.body.appendChild(nav);
}})();
</script>
""", height=0, scrolling=False)

    if st.session_state.chapters:
        st.divider()
        full_novel = "\n\n\n".join(st.session_state.chapters)
        _dl_title = (s.get("world_input", "") or "小說")[:20]
        _dl_name = s.get("name", "")
        st.download_button(
            label=f"⬇️ 下載全部章節（{len(st.session_state.chapters)} 章）",
            data=full_novel,
            file_name=f"{_dl_title}_{_dl_name}.txt",
            mime="text/plain",
        )
