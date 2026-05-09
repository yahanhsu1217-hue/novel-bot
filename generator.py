from __future__ import annotations

import json as _json
import re as _re

from openai import OpenAI

from config import DEFAULT_MODEL, LENGTH_CHARS
from training import notes_to_prompt_block


def summarize_chapter(client: OpenAI, chapter_text: str, chapter_num: int) -> tuple[str, dict]:
    """Generate chapter summary + extract story bible data. Returns (summary_str, bible_dict)."""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content":
            f"分析第 {chapter_num} 章，只輸出 JSON，不要其他文字：\n"
            f"{{\n"
            f'  "summary": "100字內條列關鍵事件、情感進展、決策結果",\n'
            f'  "banned_phrases": ["逐字列出本章出現、後續不可重複的具體語句或句型，至少5條"],\n'
            f'  "used_tropes": ["本章使用的情感或劇情套路"],\n'
            f'  "open_threads": ["尚未解決的伏筆或承諾，例如某角色說有話要說但未說"],\n'
            f'  "established_facts": ["本章確立的世界觀事實、角色能力、角色所在位置、重要物品、人物關係等，格式：實體+狀態，例如：主角目前無任何異能、A角色在B地點、主角不知道X秘密"]\n'
            f"}}\n\n{chapter_text}"
        }],
        max_tokens=800,
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    try:
        m = _re.search(r"\{[\s\S]*\}", text)
        data = _json.loads(m.group()) if m else {}
        return data.get("summary", text[:200]), data
    except Exception:
        return text[:200], {}


def extract_story_bible(client: OpenAI, chapter_text: str, chapter_num: int) -> dict:
    _, bible = summarize_chapter(client, chapter_text, chapter_num)
    return bible


def analyze_writing_style(client: OpenAI, sample_text: str) -> str:
    """Analyze a writing sample and return a style description for use in prompts."""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content":
            "請分析以下文章的寫作風格特色，輸出一段連貫說明（不要分點標題），"
            "具體描述並引用原文詞語或句子佐證，涵蓋以下面向：\n"
            "句子節奏（短促有力？長句迴旋？混合節奏？）、"
            "動詞的力度與精準度（舉代表性動詞，說明為何這些動詞有效）、"
            "形容詞密度（稀疏精準？密集堆疊？）、"
            "感官描寫方式（最常用哪種感官？如何具體呈現觸覺、溫度、氣味？）、"
            "時間過渡手法（如何處理時間流逝？是否展開過程而非跳躍？）、"
            "情緒呈現方式（直述情緒？透過肢體動作？透過環境細節？）、"
            "整體文字密度（每句話的信息量高低）。\n\n"
            "分析目標：讓另一位作者能照此密度和選字精準度寫出相似質感的文字。\n\n"
            f"{sample_text}"
        }],
        max_tokens=900,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def fix_consistency(client: OpenAI, chapter_text: str, protagonist_name: str = "", extra_names: list[str] | None = None, nickname: str = "") -> str:
    """Scan chapter for internal contradictions (location, numbers, facts) and fix them."""
    name_rules = []
    if protagonist_name:
        allowed = f"「{protagonist_name}」" + (f"或暱稱「{nickname}」" if nickname else "")
        name_rules.append(f"- 其他角色稱呼主角時，只能使用 {allowed}，若出現其他翻譯、音譯或自創名字，一律改回正確稱呼")
    for n in (extra_names or []):
        if n.strip():
            name_rules.append(f"- 角色「{n}」的名字不可被翻譯或音譯，若出現其他語言的近似詞（如中文音譯），一律改回「{n}」")
    name_rule = "\n".join(name_rules) + "\n" if name_rules else "- 所有角色名字不可被翻譯或音譯\n"
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content":
            "請仔細閱讀以下章節，找出並修正所有內部矛盾，包括：\n"
            "- 同一角色在同一段時間出現在兩個不同地點\n"
            "- 同一角色前後說出矛盾的距離、時間、數字\n"
            "- 角色在某處做了動作，下一句卻出現在另一處而未交代移動\n"
            "- 角色稱呼自己用了名字而非第一人稱\n"
            f"{name_rule}\n"
            "規則：\n"
            "1. 只修正有矛盾的地方，其餘文字完全保留，不可改動\n"
            "2. 直接輸出修正後的完整章節正文，不要加說明或標記\n\n"
            f"{chapter_text}"
        }],
        max_tokens=6000,
        temperature=0.1,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else chapter_text


_SYSTEM = """你是一位頂尖的小說作家，擅長創作沉浸感強的連載故事。

【最高優先級規則 — 必須百分之百遵守，不可違背】
- 所有角色的姓名、外貌、個性、住所、關係，嚴格按照用戶設定，不可自行更改或忽略
- 所有角色（包括主角與配角）的姓名必須全程使用用戶填寫的原始名稱，禁止翻譯、音譯、縮寫、取暱稱或改為其他語言的近似詞。例如：用戶填寫「Dalon」就必須全程寫「Dalon」，絕不可寫成「達隆」「大龍」或任何中文近似詞；填寫「Allison」就必須全程寫「Allison」，不可寫成「艾莉森」或其他中文音譯。
- 若敘事視角為第一人稱，旁白中的主角必須全程以「我」自稱，嚴禁在旁白中用主角名字作為第三人稱主語；其他角色在對話中可以叫主角的名字，但旁白只能是「我」。
- 環境設定（地點、氛圍）必須貫穿全文，不可偷換場景
- 劇情限制（禁止出現的內容）為硬性禁令，違反即為失敗
- 希望出現的劇情元素必須在本章中實際發生，不可只用旁白帶過
- 嚴禁憑空引入從未在前章建立的設定：若角色從未被描述擁有異能、特殊技術、秘密身份、特定物品，後續章節不可突然使其擁有；任何新能力、新背景、新秘密，必須先在故事中有所鋪墊或伏筆，才能揭露
- 【已確立的世界事實】區塊為硬性約束，其中每一條都不可違背，包括角色當前位置、能力狀態、人物關係現況
- CP 親密程度為整體故事的上限基調，但感情必須從零開始發展：初期兩人只是普通關係（陌生人、同事、宿敵、朋友等），不存在互相曖昧或明顯的好感；隨章節推進，才因共同經歷而逐漸產生情感，曖昧與心動只能在中後期自然浮現，不可從第一章就讓兩人互送秋波或明顯互相在意
- 角色說話方式與個性必須全程一致，不可前後矛盾

【地理與空間邏輯 — 嚴格遵守】
- 所有地點、住所、移動路線必須符合世界觀的地理邏輯，不可憑空出現不合理的場所
- 角色從 A 地前往 B 地，必須考量世界設定中合理的交通方式與距離
- 各角色的住所與其身份、世界背景相符，整個故事中保持一致，章節間不可互相矛盾
- 場景切換時需自然交代地點轉換的原因與方式，不可突兀跳場
- 末日、架空、異世界等特殊世界觀，地理規則以世界設定為準，不套用現實邏輯

【角色在場邏輯 — 嚴格遵守】
- 每個場景開始前，必須在心中確認「哪些角色在場、哪些不在場」
- 剛剛說過話或做過動作的角色，下一刻不可被其他人詢問「他/她在哪裡」，這是自相矛盾
- 角色離開或加入場景，必須有明確的交代（例如：走進房間、接到電話、突然出現），不可憑空消失或出現
- 同一場景中，不可讓同一角色同時出現在兩個不同地點
- 若角色確實不在場，其他人提到他時用「他不在這裡」「他去了…」等方式說明，而非讓不在場的人突然開口

【場景空間與動作連續性 — 嚴格遵守】
- 角色的每一個動作必須符合上一個動作的結果：關門離開後不可立刻出現在室內；走出房間後不可未經交代又在房間裡開口
- 同一段對話或決策不可在不同位置重複發生：若某個爭論已經結束並做出決定，不可再以幾乎相同的內容重演一次
- 場景內的位置描寫必須前後一致：角色站在門口，下一句不可未經移動就站在客廳中央
- 動作的因果鏈必須完整：A 發生 → B 因此發生 → C 接續，不可跳過中間步驟或讓結果與原因矛盾
- 寫每一句前先問：「這個角色上一刻在哪裡、做了什麼？這句話是否符合那個狀態？」

【對話自稱邏輯 — 嚴格遵守】
- 角色說話時絕對不可用自己的名字稱呼自己，必須用「我」「我們」
  錯誤示範：Andy說「你跟Andy走」→ 正確：Andy說「你跟我走」
- 角色說話時稱呼其他人可以用名字，但稱呼自己只能用第一人稱
- 角色下達指令或說明計畫時，「我」代表說話者本人，不可混淆
- 旁白描述和對話引號內的視角必須一致，不可在引號內突然切換人稱

【重複禁令 — 嚴格遵守】
- 禁止重複使用相同或高度相似的句子、描寫、對話，包括措辭、句型結構
- 禁止重複出現相似的情節走向：例如「角色A突然出現救場」「眾人爭論後達成共識」等模板橋段不可在同一故事中反覆使用
- 每個場景的情緒基調、衝突類型、解決方式都必須與前面章節有所區別
- 描寫同一角色的動作或反應時，不可每次都用同樣的詞彙或句型
- 寫作前必須回顧「前面章節已發生的事件摘要」，確保新內容在情節和語言上都是全新的推進

【嚴格禁止的情感套路模板 — 整個故事禁止使用，不得以任何變體出現】
以下橋段禁止出現，連結構相似的版本也不行，遇到相似情境時必須從根本上用不同敘事弧線處理：
- 「角色表態要同行」→「主角心中湧出暖流／某種感動」→「好，一起去」（整條弧線禁用）
- 角色用堅定眼神或語氣說出一句話，主角因此被打動後改變態度
- 爭執後沉默，其中一人主動示好，另一人立刻原諒或和解
- 主角獨自出發，某人在身後叫住，說出感人或堅定的話
- 危機解除後相視而笑、互相擁抱、說「我們做到了」
- 角色說「你別想甩掉我」或語義相近的任何句子
- 角色說出感人宣言後，另一角色沉默、眼眶泛紅、或點頭
- 心跳加速 + 移開視線 + 沒說話 的曖昧三件套
【場景結構多樣性 — 強制要求】
在寫任何情感或衝突場景前，必須先確認：這個場景的起點、衝突類型、化解方式，和前面章節是否有本質上的不同。
若起點相似（如「兩人意見分歧」），化解方式必須截然不同：前一章用了「沉默後和解」，本章必須用「衝突升級、其中一人離開」或「條件談判、各退一步但心存芥蒂」或「根本沒有解決，帶著裂痕繼續」。
禁止每次危機都能靠一句話或一個眼神化解。

【創作守則】
1. 世界觀、規則與邏輯嚴格遵守，不可自創矛盾設定
2. 連載時每章情節連貫，有明確起伏與鉤子帶入下一章
3. 細節豐富，對話自然，場景描寫生動
4. 嚴格按照指定字數撰寫
5. 開始寫作前，在心中默默核對每一條設定是否都已納入"""


def _world_str(world_mode: str, world_input: str) -> str:
    if world_mode == "作品世界":
        return f"作品：《{world_input}》\n（請嚴格還原原作世界觀、角色性格與劇情邏輯）"
    return f"世界設定：{world_input}"


_STYLE_SYSTEM_ADDON = """
【文風寫作規則 — 強制執行，整篇必須貫徹，不可違背】
- 嚴禁使用模糊或抽象的詞彙；每個感受必須用具體、可感知的詞語表達（觸覺、溫度、氣味、聲音、視覺細節），禁止寫「難以言說的感覺」「某種情緒湧上」
- 重要場景不可用一句話帶過；必須展開描寫，包含觸覺、溫度、氣味、節奏感等細節
- 凡覺得可以略過之處，正是必須展開的地方；嚴禁用省略或暗示跳過任何場景
- 嚴禁使用「隨後」「不久後」「過了一會兒」「稍後」「片刻後」「沒多久」等跳躍時間的詞略過過程；必須寫出過程本身
- 情緒反應必須透過身體感受呈現（心跳、呼吸變化、皮膚反應、肌肉張力、手的動作），而非直接陳述「她感到緊張」「他心情複雜」
- 動詞選擇有力且精準，避免「走」「說」「看」「感到」等平淡動詞，改用具體動作的精確描述
"""

_NSFW_ADDON = """
【限制級模式 — 強制執行】
本故事已由用戶明確開啟限制級模式，你必須按照以下規則執行，不可以保守或迴避的方式處理親密場景：
- 當劇情發展到親密時機，必須完整描寫性愛場景，包含肢體細節、感官描寫、情緒反應，不可用省略號、「XXX」、「此處略去」、「兩人相擁」等方式跳過
- 禁止用「難以言說的歡愉」「沉入彼此」「時間凝固」等模糊帶過；必須具體、直接、細膩地描寫發生的事
- 筆觸應兼顧情感與肉體，有層次、有溫度，非機械羅列
- 場景需符合雙方自願
- 親密時機由劇情自然決定，不可每章強插，但到了該發生的時刻，必須完整呈現，不可縮水
"""

_INTIMACY_GUIDE = {
    "清水（純愛暗戀）": (
        "以暗戀與心動為主，描寫心跳加速、眼神交會、不敢靠近的曖昧。"
        "可以有牽手或短暫的肢體接觸，但保持純情克制。"
    ),
    "糖（甜蜜互動）": (
        "積極描寫兩人的甜蜜互動：擁抱、依偎、額頭相碰、溫柔凝視。"
        "要有明確的親密場景，讓讀者感受到兩人之間的溫度與愛意。"
        "對話要帶有曖昧或直白的情感表達。"
    ),
    "甜虐（曖昧張力）": (
        "在甜蜜中加入張力與衝突：欲言又止的告白、身體距離極近的對峙、"
        "突如其來的吻或被打斷的吻、誤解與和解。情緒起伏大，讓讀者又甜又虐。"
    ),
    "熾熱（激情親密）": (
        "描寫強烈的情感爆發與肢體親密：深吻、緊擁、情感失控的瞬間。"
        "場景要有熱度與張力，細節描寫情感與肢體語言，讓親密感真實可感。"
    ),
}


def _cp_str(cp_type: str, cp_characters: list[dict], name: str, nsfw: bool = False) -> str:
    if cp_type == "我 × 角色" and cp_characters:
        valid = [c for c in cp_characters if c.get("name", "").strip()]
        if not valid:
            return "CP 設定：無 CP，純故事向"
        display_names = "、".join(
            "（AI 自選）" if c["name"].strip() == "隨機" else c["name"]
            for c in valid
        )
        lines = [f"CP 設定：我（{name}）× {display_names}"]
        for c in valid:
            char_name = c["name"].strip()
            intimacy = c.get("intimacy", "糖（甜蜜互動）")
            guide = _INTIMACY_GUIDE.get(intimacy, _INTIMACY_GUIDE["糖（甜蜜互動）"])
            if nsfw:
                guide += "（限制級模式：後期必須包含明確的性愛場景，完整描寫肢體接觸與感官細節，不可迴避或省略）"
            if char_name == "隨機":
                lines.append(f"- AI 自選角色：請從世界中挑選合適角色，親密程度上限：{guide}")
            else:
                lines.append(f"- {char_name}：親密程度上限：{guide}")
        lines += [
            "【感情發展節奏 — 嚴格遵守】",
            "- 故事開始時與所有 CP 對象之間沒有任何曖昧或特別的好感，關係從普通甚至疏遠開始",
            "- 前期（前 1/3）：只是普通互動，也許有輕微的注意，但絕對不能有明顯的心動或曖昧行為",
            "- 中期（中 1/3）：因共同經歷開始有更深的了解，可能出現一方先有感覺，但仍壓抑或不確定",
            "- 後期（後 1/3）：感情才逐漸浮現，曖昧與心動自然流露",
            "- 禁止在前期安排：深情凝視、心跳加速、互送秋波、曖昧對話、明顯的互相在意",
        ]
        return "\n".join(lines)
    return "CP 設定：無 CP，純故事向"


def _extra_chars_str(chars: list[dict]) -> str:
    valid = [c for c in chars if c.get("name", "").strip()]
    if not valid:
        return ""
    lines = ["【其他角色】（以下所有名字為原始設定，不可翻譯或音譯）"]
    for c in valid:
        parts = [c["name"]]
        if c.get("appearance"):   parts.append(f"外貌：{c['appearance']}")
        if c.get("personality"):  parts.append(f"個性：{c['personality']}")
        if c.get("residence"):    parts.append(f"住所：{c['residence']}")
        if c.get("skills"):       parts.append(f"技能：{c['skills']}")
        if c.get("relationship"): parts.append(f"與我的關係：{c['relationship']}")
        lines.append("- " + "，".join(parts))
    return "\n".join(lines)


def _must_appear_block(extra_characters: list[dict], cp_characters: list[dict], cp_type: str) -> str:
    seen: set[str] = set()
    names: list[str] = []
    for c in extra_characters:
        n = c.get("name", "").strip()
        if n and n not in seen:
            seen.add(n)
            names.append(n)
    if cp_type == "我 × 角色":
        for c in (cp_characters or []):
            n = c.get("name", "").strip()
            if n and n != "隨機" and n not in seen:
                seen.add(n)
                names.append(n)
    if not names:
        return ""
    return (
        "【必須出場的角色 — 強制執行】\n"
        "以下每位角色本章必須實際出現，有對話或具體行動，不可只被提及或完全缺席：\n"
        + "\n".join(f"- {n}" for n in names)
    )


def stream_chapter(
    client: OpenAI,
    world_mode: str,
    world_input: str,
    language: str,
    perspective: str,
    length_label: str,
    cp_type: str,
    cp_characters: list[dict],
    name: str,
    nickname: str,
    gender: str,
    personality: str,
    appearance: str,
    background: str,
    extra_characters: list[dict],
    chapter_num: int,
    prev_chapter_text: str = "",
    plot_want: str = "",
    plot_forbid: str = "",
    total_chapters: int = 5,
    is_final: bool = False,
    character_notes: str = "",
    prev_summaries: list[str] | None = None,
    environment: str = "",
    residence: str = "",
    story_bible: dict | None = None,
    nsfw: bool = False,
    training_notes: list[dict] | None = None,
    chapter_directive: str = "",
    style_reference: str = "",
):
    target = LENGTH_CHARS.get(length_label, 2000)

    plot_block = "\n".join(filter(None, [
        f"【希望出現的劇情元素】\n{plot_want}" if plot_want else "",
        f"【嚴格禁止出現的內容】\n{plot_forbid}" if plot_forbid else "",
    ]))

    notes_block = (
        f"【原作角色外貌與設定（以下為準，不可違背，優先於你的記憶）】\n{character_notes}"
        if character_notes else ""
    )

    history_block = ""
    if prev_summaries:
        lines = "\n".join(f"第 {i+1} 章：{s}" for i, s in enumerate(prev_summaries))
        history_block = (
            f"【前面章節記錄（新章節必須與以下內容完全不同）】\n"
            f"嚴格禁止：重複相同情節走向、橋段類型、對話模式、描寫詞彙、句型結構。\n"
            f"{lines}"
        )

    env_block = f"【環境設定】\n{environment}" if environment else ""

    bible_block = ""
    if story_bible:
        parts = []
        if story_bible.get("banned_phrases"):
            phrases = "\n".join(f"- {p}" for p in story_bible["banned_phrases"])
            parts.append(f"【絕對禁止重複使用的語句（逐字禁用）】\n{phrases}")
        if story_bible.get("used_tropes"):
            tropes = "\n".join(f"- {t}" for t in story_bible["used_tropes"])
            parts.append(
                f"【已用過的套路 — 絕對禁止重複，連結構相似的版本都不行】\n"
                f"以下套路已出現，後續章節處理相似情境時，必須從敘事結構根本上做出改變：\n"
                f"{tropes}\n"
                f"寫作前必須自問：「這個場景的情感弧線和以上哪條最相似？如果相似，我要怎麼從起點就走一條截然不同的路？」"
            )
        if story_bible.get("open_threads"):
            threads = "\n".join(f"- {t}" for t in story_bible["open_threads"])
            parts.append(f"【未解決的伏筆（必須在後續章節回應）】\n{threads}")
        if story_bible.get("established_facts"):
            facts = "\n".join(f"- {f}" for f in story_bible["established_facts"])
            parts.append(
                f"【已確立的世界事實（絕對不可違背或自相矛盾）】\n"
                f"以下是前面章節已明確建立的事實，後續章節不得推翻、忽略或與之矛盾：\n{facts}"
            )
        if parts:
            bible_block = "\n".join(parts)

    # Build geography summary for logical consistency
    geo_lines = ["【地理設定摘要（所有場景移動必須符合此設定）】"]
    if environment:
        geo_lines.append(f"故事環境：{environment}")
    if residence:
        geo_lines.append(f"我（{name}）的住所：{residence}")
    for c in extra_characters:
        if c.get("name") and c.get("residence"):
            geo_lines.append(f"{c['name']} 的住所：{c['residence']}")
    geo_block = "\n".join(geo_lines) if len(geo_lines) > 1 else ""

    me_line = "，".join(filter(None, [
        f"主角（我）：{name}，{gender}",
        f"外貌：{appearance}",
        f"個性：{personality}",
        f"住所：{residence}" if residence else "",
    ]))

    # Name reference table — prevents model from translating character names
    name_ref_lines = [f"【角色名字對照表 — 以下名字全程原樣使用，禁止翻譯或音譯】"]
    if name:
        name_ref_lines.append(f"- 主角：{name}（唯一合法寫法，不可有任何其他形式）")
    for c in extra_characters:
        n = c.get("name", "").strip()
        if n:
            name_ref_lines.append(f"- {n}（唯一合法寫法，不可有任何其他形式）")
    name_ref_block = "\n".join(name_ref_lines) if len(name_ref_lines) > 1 else ""

    training_block = notes_to_prompt_block(training_notes or [])

    style_block = ""
    if style_reference:
        style_block = (
            "【文風參考分析 — 最高優先級，用相同的密度、節奏與選字精準度寫作】\n"
            f"{style_reference}\n\n"
            "【強制執行的具體規則】\n"
            "- 嚴禁模糊或抽象詞彙；每個感受用具體可感知的詞語表達（觸覺、溫度、氣味、聲音）\n"
            "- 重要場景必須展開描寫，禁止一句帶過\n"
            "- 凡覺得可以略過之處，正是必須展開的地方\n"
            "- 嚴禁「隨後」「不久後」「過了一會兒」「稍後」「片刻後」等跳躍時間的詞\n"
            "- 情緒必須透過身體感受呈現，而非直接陳述情緒\n"
            "- 動詞有力精準，避免「走」「說」「看」「感到」等平淡動詞"
        )

    setting_block = "\n".join(filter(None, [
        style_block,
        training_block,
        _world_str(world_mode, world_input),
        env_block,
        geo_block,
        notes_block,
        name_ref_block,
        _cp_str(cp_type, cp_characters, name, nsfw),
        me_line,
        _extra_chars_str(extra_characters),
        _must_appear_block(extra_characters, cp_characters, cp_type),
        plot_block,
        history_block,
        bible_block,
    ]))

    progress_note = ""
    if is_final:
        progress_note = "【注意】這是最終章，請將所有伏筆收束，給出完整且令人滿足的結局，不留懸念。"
    elif total_chapters > 1:
        remaining = total_chapters - chapter_num
        if remaining == 1:
            progress_note = f"【進度提示】這是倒數第二章（共 {total_chapters} 章），請開始收攏主線衝突，為結局鋪墊。"
        else:
            progress_note = f"【進度提示】目前第 {chapter_num} 章，共規劃 {total_chapters} 章，請根據進度調整敘事節奏。"

    is_first_person = "第一人稱" in perspective
    _call_names = "、".join(filter(None, [name, nickname]))
    pov_instruction = (
        f"敘事視角：第一人稱。旁白全程以「我」敘述，絕對禁止在旁白中用主角名字作為第三人稱主語。"
        f"其他角色對主角說話時，只能使用以下稱呼：{_call_names}，不可自創其他名字或音譯。"
        f"旁白中的主角只能用「我」。"
        if is_first_person else
        f"敘事視角：第三人稱。旁白以主角名字「{name}」敘述。"
    )

    directive_block = (
        f"【本章特別指示 — 最高優先級，必須在本章中執行】\n{chapter_directive}"
        if chapter_directive.strip() else ""
    )

    if chapter_num == 1:
        prompt = f"""{setting_block}

輸出語言：{language}
{pov_instruction}
目標字數：{target} 字
{progress_note}
{directive_block}

【我的背景故事】
{background}

請創作第一章，建立世界氛圍與角色，帶出故事開端{"，給出完整結局。" if is_final else "，結尾留下讓人想繼續讀的鉤子。"}
直接輸出故事正文，格式如下：

第一章　[章節標題]

（正文）"""
    else:
        context = prev_chapter_text[-2500:] if len(prev_chapter_text) > 2500 else prev_chapter_text
        ending_instruction = (
            "請將故事帶向完整結局，收束所有主線與情感線，給讀者滿足感。"
            if is_final else
            "請自然銜接上一章，推進情節，帶出新的發展或衝突，結尾留下鉤子。"
        )
        prompt = f"""故事設定：
{setting_block}

輸出語言：{language}
{pov_instruction}
目標字數：{target} 字
{progress_note}
{directive_block}

【上一章結尾】
{context}

{ending_instruction}
直接輸出故事正文，格式如下：

第{chapter_num}章　[章節標題]

（正文）"""

    training_block_sys = notes_to_prompt_block(training_notes or [])
    system_content = (
        _SYSTEM
        + (_NSFW_ADDON if nsfw else "")
        + (_STYLE_SYSTEM_ADDON if style_reference else "")
        + (f"\n\n{training_block_sys}" if training_block_sys else "")
    )

    stream = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=6000,
        temperature=0.72,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
