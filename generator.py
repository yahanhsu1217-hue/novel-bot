from __future__ import annotations

import json as _json
import re as _re
# v2

from openai import OpenAI

from config import DEFAULT_MODEL, GEMINI_MODEL, DEEPSEEK_MODEL, GEMINI_BASE_URL, LENGTH_CHARS
from training import notes_to_prompt_block


def _model_for(client: OpenAI) -> str:
    """Pick the right model name based on which client is being used."""
    try:
        if GEMINI_BASE_URL.rstrip("/") in str(client.base_url):
            return GEMINI_MODEL
    except Exception:
        pass
    return DEEPSEEK_MODEL


def summarize_chapter(client: OpenAI, chapter_text: str, chapter_num: int) -> tuple[str, dict]:
    """Generate chapter summary + extract story bible data. Returns (summary_str, bible_dict)."""
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            f"分析第 {chapter_num} 章，只輸出 JSON，不要其他文字：\n"
            f"{{\n"
            f'  "summary": "300字內條列（省略性愛場景細節，只記主線）：①關鍵事件與結果 ②角色當前位置與狀態 ③情感進展與關係變化 ④尚未解決的衝突或伏筆 ⑤重要決策與後果",\n'
            f'  "banned_phrases": ["逐字列出本章出現、後續不可重複的具體語句或句型，至少5條"],\n'
            f'  "used_tropes": ["本章使用的情感或劇情套路"],\n'
            f'  "open_threads": ["尚未解決的伏筆或承諾，例如某角色說有話要說但未說"],\n'
            f'  "asked_questions": ["本章中任何角色問過主角的所有問題，逐條概括，例如：你有喜歡的人嗎、你養過寵物嗎、你去過哪些地方、你的家人是做什麼的——哪怕是閒聊性質的問題也要列出"],\n'
            f'  "established_facts": ["本章確立的所有事實，必須涵蓋以下四類，格式：實體+狀態——'
            f'①世界觀與環境：地點規則、重要物品、角色所在位置；'
            f'②角色能力與經歷：技能、異能、職業、過去經歷；'
            f'③主角個人自我揭露：主角在對話或旁白中說過的關於自己的任何個人資訊，例如：主角說自己沒養過寵物、主角說媽媽對毛過敏、主角說自己目前沒有喜歡的人、主角說自己從未去過某地——這類事實必須逐條列出，不可省略；'
            f'④人物關係現況：誰知道什麼秘密、誰對誰有什麼感受、已確認的關係狀態"]\n'
            f"}}\n\n{chapter_text}"
        }],
        max_tokens=1600,
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


def outline_from_chapter(client: OpenAI, chapter_text: str, chapter_num: int) -> str:
    """Read an existing chapter and produce a concise outline (150-250 chars) describing what happened."""
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            f"請為以下第 {chapter_num} 章內容，用 150–250 字寫一段大綱摘述。"
            f"格式：條列重點事件與情感進展，保留關鍵場景與轉折，不加標題。\n\n{chapter_text}"
        }],
        max_tokens=400,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def compress_old_summaries(
    client: OpenAI,
    existing_overview: str,
    old_summaries: list[str],
    chapter_start: int = 1,
) -> str:
    """Merge old chapter summaries into a concise early-story overview. Omits NSFW details."""
    if not old_summaries and not existing_overview:
        return ""
    parts = []
    if existing_overview:
        parts.append(f"【現有早期總覽】\n{existing_overview}")
    if old_summaries:
        lines = "\n".join(f"第 {chapter_start + i} 章：{s}" for i, s in enumerate(old_summaries))
        parts.append(f"【待合併章節摘要】\n{lines}")
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            "請將以下早期章節摘要合併為一段簡潔的「早期故事總覽」，供後續章節生成時參考。\n\n"
            "規則：\n"
            "- 只保留：主線情節走向、角色當前位置與狀態、重要關係變化、關鍵決策與後果、尚未解決的伏筆\n"
            "- 完全省略：性愛場景細節、親密接觸的具體描寫；若有相關情節，只寫「兩人關係進一步發展」等概括詞\n"
            "- 輸出格式：條列式，每項以「・」開頭，總長約 200-400 字\n"
            "- 直接輸出總覽內容，不加標題、前言或說明\n\n"
            + "\n\n".join(parts)
        }],
        max_tokens=800,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def analyze_writing_style(client: OpenAI, sample_text: str) -> str:
    """Analyze a writing sample and return actionable style rules with concrete examples."""
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            "請分析以下文章的寫作風格，輸出可直接讓另一位作者照做的具體規則清單。\n"
            "每條規則必須：①說明做法 ②從原文引用一個實際句子作為示範。\n\n"
            "涵蓋以下面向（每項都要有原文引用）：\n"
            "1. 句子長度與標點密度：這篇文章的句子平均多長？怎麼斷句？引用一句最有代表性的。\n"
            "2. 動詞選擇：這篇用了哪些有力的動詞？引用兩到三個，說明它們比平淡動詞強在哪裡。\n"
            "3. 感官描寫：這篇怎麼呈現觸覺/溫度/聲音？引用一句感官描寫句，說明它具體在哪裡。\n"
            "4. 情緒呈現：角色情緒怎麼表達？直述？肢體？環境？引用一句示範。\n"
            "5. 時間推進：場景如何從A過渡到B？引用一個時間過渡的寫法。\n"
            "6. 整體禁忌：這篇明顯迴避了哪些寫法（例如從不直接說「她感到緊張」、從不用「那股」開頭）？\n\n"
            "格式：每一條直接給規則+引用，不要加長篇解釋。目的是讓模型模仿，不是讓人讀懂。\n\n"
            f"{sample_text}"
        }],
        max_tokens=1200,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def fix_repetitive_paragraphs(client: OpenAI, chapter_text: str, preserve_ending: bool = False) -> str:
    """Detect verbatim repeated paragraph blocks and rewrite the duplicates."""
    paragraphs = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]
    seen: dict = {}
    has_duplicates = False
    for para in paragraphs:
        key = para[:40]
        if key in seen:
            has_duplicates = True
            break
        seen[key] = 1
    if not has_duplicates:
        return chapter_text
    ending_rule = (
        "- 【結尾保護】最後一段是作者設定的強制結尾，嚴禁改寫、刪除或在其後添加任何文字\n"
        if preserve_ending else ""
    )
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            "以下章節中存在逐字重複的段落（同一段文字出現兩次以上）。\n"
            "請找出所有重複的段落，將第二次及之後出現的改寫成推進場景的新內容：\n"
            "- 動作要有變化：力道、節奏、角度、深度、速度至少改變一項\n"
            "- 兩人的反應要有層次推進：情緒升溫、身體變化、對話內容演進\n"
            "- 不可只換幾個詞，必須寫出真正不同的場景發展\n"
            "- 第一次出現的段落完全保留不動\n"
            f"{ending_rule}"
            "- 直接輸出修改後的完整章節正文，不加任何說明或標記\n"
            "- 必須輸出完整全文，不可截斷或省略；嚴禁在原文最後一句之後添加任何新文字\n\n"
            f"{chapter_text}"
        }],
        max_tokens=8000,
        temperature=0.5,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else chapter_text


def fix_sensory_crutches(client: OpenAI, chapter_text: str, preserve_ending: bool = False) -> str:
    """Rewrite overused 她能X到那股Y sensory perception sentences into direct descriptions."""
    import re
    count = len(re.findall(r'她能[^\s，。！？]{1,4}到', chapter_text))
    if count <= 3:
        return chapter_text
    ending_rule = (
        "- 【結尾保護】最後一段是作者設定的強制結尾，嚴禁改寫或在其後添加任何文字\n"
        if preserve_ending else ""
    )
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            f"以下章節中，「她能X到」（如「她能感覺到」「她能聽到」「她能聞到」「她能看到」）的句式出現了 {count} 次，嚴重超標。\n"
            "請找出所有這類句子，逐一改寫為直接以名詞或動詞切入的句子，不透過感知動詞中繼。\n\n"
            "改寫原則：\n"
            "- 以發出感覺的物體、聲音、氣味、動作本身為主詞，直接描述它的狀態\n"
            "- ✗「她能感覺到那股濕潤——愛液在擴散」→ ✓「愛液粘在皮膚上，涼的，帶著腥甜」\n"
            "- ✗「她能聽到那股聲音——腳步聲傳來」→ ✓「腳步聲從走廊傳來，很輕，刻意壓低的」\n"
            "- ✗「她能感覺到那股視線——Dalon Tsai的目光落在她身上」→ ✓「Dalon Tsai的目光停在她身上，不動」\n"
            "- 每句改寫後必須保留原句的信息，不可刪除內容\n"
            "- 只改「她能X到」的句子，其餘文字完全不動\n"
            f"{ending_rule}"
            "- 直接輸出修改後的完整章節正文，不加任何說明或標記\n"
            "- 必須輸出完整全文，不可截斷；嚴禁在原文最後一句之後添加任何新文字\n\n"
            f"{chapter_text}"
        }],
        max_tokens=8000,
        temperature=0.3,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else chapter_text


def fix_range_patterns(client: OpenAI, chapter_text: str, preserve_ending: bool = False) -> str:
    """Rewrite overused 從X到X / 從X蔓延到Y sentence-structure patterns."""
    import re
    count = len(re.findall(r'從[^，。！？\n]{1,20}(?:到|變成|轉變為|蔓延到|延伸到|擴散到)[^，。！？\n]{1,25}', chapter_text))
    if count <= 2:
        return chapter_text
    ending_rule = (
        "- 【結尾保護】最後一段是作者設定的強制結尾，嚴禁改寫或在其後添加任何文字\n"
        if preserve_ending else ""
    )
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            f"以下章節中，「從X到Y」「從X變成Y」「從X蔓延到Y」「從X延伸到Y」等狀態推進句型出現了 {count} 次，嚴重超標（上限為 2 次）。\n\n"
            "需要找出並改寫的模式包括（全部必須處理）：\n"
            "① 連續鏡像翻轉：「從A到B，從B到A」→ 只保留一個方向，另一個改寫\n"
            "② 連續遞進：「從A變成B，從B變成C」「從A到B，從B到C」→ 直接寫最終狀態 C\n"
            "③ 相同骨架在同一段重複出現兩次以上\n\n"
            "改寫原則：\n"
            "- 整章保留最多 2 個「從X到X」句，其餘全部改寫\n"
            "- 改用結果狀態直接呈現：「腿部肌肉發硬」而非「從大腿到腹部蔓延」\n"
            "- 改用動作主詞切入：「腰肢不受控地扭動」「腹肌繃緊收縮」\n"
            "- 改用聲音或觸感直接切入：「布料間擠出細微的水聲」\n"
            "- 只改「從X到X」骨架的句子，其餘文字完全不動\n"
            f"{ending_rule}"
            "- 直接輸出修改後的完整章節正文，不加任何說明或標記\n"
            "- 必須輸出完整全文，不可截斷\n\n"
            f"{chapter_text}"
        }],
        max_tokens=8000,
        temperature=0.5,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else chapter_text


def fix_nagu_patterns(client: OpenAI, chapter_text: str, preserve_ending: bool = False) -> str:
    """Rewrite overused 那股X sentence-opening patterns."""
    import re
    count = len(re.findall(r'那股[^，。！？\n]{1,20}', chapter_text))
    if count <= 4:
        return chapter_text
    ending_rule = (
        "- 【結尾保護】最後一段是作者設定的強制結尾，嚴禁改寫或在其後添加任何文字\n"
        if preserve_ending else ""
    )
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            f"以下章節中，「那股X」句型（如「那股觸感」「那股濕熱感」「那股溫熱」「那股心跳」）出現了 {count} 次，嚴重超標（上限為 4 次）。\n\n"
            "「那股X」是一種中繼結構：先命名感受，再描述它，等於描述了兩次。\n"
            "請找出超標的部分，改用直接描述法：\n\n"
            "改寫原則：\n"
            "- 以感受的來源當主詞，直接描述動作或狀態，不用「那股」作為過渡\n"
            "- ✗「那股觸感透過布料傳來——溫熱的，帶著黏性」\n"
            "  ✓「布料貼著皮膚，溫熱的，帶著黏性」\n"
            "- ✗「那股濕熱感從陰道深處滲出」\n"
            "  ✓「體液從陰道深處滲出，溫熱的，在皮膚上擴散」\n"
            "- ✗「那股心跳透過兩層布料傳來——規律的，穩定的」\n"
            "  ✓「心跳從胸口傳過來，規律的，穩定的」\n"
            "- 整章保留最多 4 個「那股」，其餘全部改寫\n"
            "- 只改「那股X」句，其餘文字完全不動\n"
            f"{ending_rule}"
            "- 直接輸出修改後的完整章節正文，不加任何說明或標記\n"
            "- 必須輸出完整全文，不可截斷\n\n"
            f"{chapter_text}"
        }],
        max_tokens=8000,
        temperature=0.4,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else chapter_text


def fix_passive_negations(client: OpenAI, chapter_text: str, preserve_ending: bool = False) -> str:
    """Rewrite overused 她沒有X / 角色名沒有X standalone negation sentences."""
    import re
    count = len(re.findall(r'(?:她|他|若渝|林澄夏|[^\s，。！？\n]{2,4})沒有[^，。！？\n]{1,12}[。]', chapter_text))
    if count <= 3:
        return chapter_text
    ending_rule = (
        "- 【結尾保護】最後一段是作者設定的強制結尾，嚴禁改寫或在其後添加任何文字\n"
        if preserve_ending else ""
    )
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            f"以下章節中，「她沒有X。」「他沒有X。」「角色名沒有X。」等否定靜止句型出現了 {count} 次，嚴重超標（上限為 3 次）。\n\n"
            "這類句子用來表達角色的無聲允許或被動反應，但重複出現會失去張力。\n"
            "請找出超過的部分，用「描寫發生了什麼」來替代「描寫沒發生什麼」：\n\n"
            "改寫原則：\n"
            "- 「她沒有退開」→ 描寫她的身體保持靜止的具體狀態，或她在做什麼小動作（調整重心、眼睛看向別處）\n"
            "- 「她沒有說話」→ 描寫她的表情、呼吸、或環境聲音填補的沉默\n"
            "- 「她沒有阻止」→ 描寫她的手的位置、肌肉的鬆弛、或她的視線落在哪裡\n"
            "- 「她沒有推開」→ 直接跳到下一個動作，讓讀者從上下文理解她的態度\n"
            "- 整章保留最多 3 個此類句子，其餘全部改寫\n"
            "- 只改否定靜止句，其餘文字完全不動\n"
            f"{ending_rule}"
            "- 直接輸出修改後的完整章節正文，不加任何說明或標記\n"
            "- 必須輸出完整全文，不可截斷\n\n"
            f"{chapter_text}"
        }],
        max_tokens=8000,
        temperature=0.5,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else chapter_text


def _para_key(text: str) -> str:
    """Normalize paragraph opening for comparison: strip punctuation/spaces, keep content chars."""
    import re
    cleaned = re.sub(r'[\s，。！？、；：「」『』【】…——\-\(\)（）\[\]〔〕""'']', '', text)
    return cleaned[:38]


def extract_paragraph_starters(chapter_text: str, min_len: int = 12) -> list[str]:
    """Extract the opening ~55 chars of each substantive paragraph for use as banned starters."""
    starters = []
    for para in chapter_text.split('\n\n'):
        para = para.strip()
        if not para or len(para) < min_len:
            continue
        first_line = para.split('\n')[0].strip()
        if first_line.startswith('第') and '章' in first_line[:8]:
            continue
        starter = para[:55].strip()
        if len(starter) >= min_len:
            starters.append(starter)
    return starters


def fix_cross_chapter_repetition(client: OpenAI, new_chapter: str, prev_chapters: list[str], nsfw: bool = False) -> str:
    """Detect paragraphs in new_chapter that duplicate any paragraph from all previous chapters and rewrite them."""
    new_paras = [p.strip() for p in new_chapter.split('\n\n') if p.strip() and len(p.strip()) > 30]
    prev_para_keys: set[str] = set()
    for ch in prev_chapters:
        for p in ch.split('\n\n'):
            p = p.strip()
            if p and len(p) > 30:
                prev_para_keys.add(_para_key(p))

    dup_keys: list[str] = []
    for para in new_paras:
        if _para_key(para) in prev_para_keys:
            dup_keys.append(para[:100])

    if not dup_keys:
        return new_chapter

    dup_list = "\n".join(f"- {d}…" for d in dup_keys[:15])
    nsfw_rule = (
        "\n- 【限制級】改寫後必須維持原段落的明確程度與細節密度，不可將情色描寫改淡或省略；"
        "身體部位仍用具體詞彙，感官細節仍須完整"
        if nsfw else ""
    )
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            "以下新章節中，有些段落與前面章節幾乎完全一致（開頭相同），必須改寫。\n\n"
            f"【重複段落開頭清單（以下每一條開頭的段落都必須改寫）】\n{dup_list}\n\n"
            "改寫規則：\n"
            "- 只改寫與前章重複的段落，其餘段落完全保留、逐字不動\n"
            "- 改寫必須用全新角度、不同細節、不同句型，不可只換幾個詞\n"
            "- 改寫後必須與上下文連貫，內容推進方向不變\n"
            f"- 直接輸出完整章節正文，不加任何說明或標記{nsfw_rule}\n"
            "- 必須輸出完整全文，不可截斷\n\n"
            f"{new_chapter}"
        }],
        max_tokens=8000,
        temperature=0.6,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else new_chapter


def fix_consistency(client: OpenAI, chapter_text: str, protagonist_name: str = "", extra_names: list[str] | None = None, nickname: str = "", preserve_ending: bool = False) -> str:
    """Scan chapter for internal contradictions (location, numbers, facts) and fix them."""
    name_rules = []
    if protagonist_name:
        allowed = f"「{protagonist_name}」" + (f"或暱稱「{nickname}」" if nickname else "")
        name_rules.append(f"- 其他角色稱呼主角時，只能使用 {allowed}，若出現其他翻譯、音譯或自創名字，一律改回正確稱呼")
    for n in (extra_names or []):
        if n.strip():
            name_rules.append(f"- 角色「{n}」的名字不可被翻譯或音譯，若出現其他語言的近似詞（如中文音譯），一律改回「{n}」")
    name_rule = "\n".join(name_rules) + "\n" if name_rules else "- 所有角色名字不可被翻譯或音譯\n"
    ending_rule = (
        "4. 【結尾保護 — 最高優先級】最後一段是作者設定的強制結尾，即使看起來突然也屬正常，"
        "嚴禁補充、延伸或在其後添加任何新對話、動作、旁白；輸出必須與原文最後一句完全一致\n"
        if preserve_ending else ""
    )
    resp = client.chat.completions.create(
        model=_model_for(client),
        messages=[{"role": "user", "content":
            "請仔細閱讀以下章節，找出並修正所有內部矛盾，包括：\n"
            "- 同一角色在同一段時間出現在兩個不同地點\n"
            "- 同一角色前後說出矛盾的距離、時間、數字\n"
            "- 角色在某處做了動作，下一句卻出現在另一處而未交代移動\n"
            "- 角色稱呼自己用了名字而非第一人稱\n"
            "- 同一角色對自身能力或技能前後矛盾（例：前面說「我會騎機車」或「車庫裡有機車」，後面卻說「我從沒騎過機車」——必須統一為前面先出現的說法）\n"
            "- 同一角色對自身擁有的物品前後矛盾（例：前面說「我有一把槍」，後面卻說「我手上沒有武器」）\n"
            "- 同一角色的記憶或經歷前後矛盾（例：前面提到認識某人，後面卻說從未見過他）\n"
            "- 同一角色的知識狀態前後矛盾（例：前面已得知某秘密，後面卻對該秘密一無所知）\n"
            "- 角色透過對話、訊息或任何方式已得知的具體資訊（如價格、約定、計畫、指示），後續場景不可讓同一角色再詢問同一件事（例：前段訊息已告知餐費並約定付款方式，後段不可讓主角再問價格——這是連續性錯誤，必須修正）\n"
            f"{name_rule}\n"
            "規則：\n"
            "1. 只修正有矛盾的地方，其餘文字完全保留，不可改動\n"
            "2. 矛盾發生時，以章節中**先出現**的說法為準，修正後出現的矛盾描述\n"
            "3. 直接輸出修正後的完整章節正文，不要加說明或標記\n"
            "4. 必須輸出完整全文，不可截斷、省略或縮短任何段落；嚴禁在原文最後一句之後添加任何新文字\n"
            f"{ending_rule}\n"
            f"{chapter_text}"
        }],
        max_tokens=8000,
        temperature=0.1,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else chapter_text


_SYSTEM = """你是一位頂尖的小說作家，擅長創作沉浸感強的連載故事。

【最高優先級規則 — 必須百分之百遵守，不可違背】
- 所有角色的姓名、外貌、個性、住所、關係，嚴格按照用戶設定，不可自行更改或忽略
- 所有角色（包括主角與配角）的姓名必須全程使用用戶填寫的原始名稱，禁止翻譯、音譯、縮寫、取暱稱或改為其他語言的近似詞。例如：用戶填寫「Dalon」就必須全程寫「Dalon」，絕不可寫成「達隆」「大龍」或任何中文近似詞；填寫「Allison」就必須全程寫「Allison」，不可寫成「艾莉森」或其他中文音譯。
- 若敘事視角為第一人稱，旁白中的主角必須全程以「我」自稱，嚴禁在旁白中用主角名字作為第三人稱主語；其他角色在對話中可以叫主角的名字，但旁白只能是「我」。
- 若敘事視角為第三人稱，旁白中絕對禁止出現「我」作為主語來描述主角；旁白必須全程以主角名字稱呼主角；「我」只允許出現在角色的對話引號之內，旁白一律用名字。違反即失敗。
- 環境設定（地點、氛圍）必須貫穿全文，不可偷換場景
- 劇情限制（禁止出現的內容）為硬性禁令，違反即為失敗
- 希望出現的劇情元素必須在本章中實際發生，不可只用旁白帶過
- 嚴禁憑空引入從未在前章建立的設定：若角色從未被描述擁有異能、特殊技術、秘密身份、特定物品，後續章節不可突然使其擁有；任何新能力、新背景、新秘密，必須先在故事中有所鋪墊或伏筆，才能揭露
- 【已確立的世界事實】區塊為硬性約束，其中每一條都不可違背，包括角色當前位置、能力狀態、人物關係現況
- CP 親密程度為整體故事的上限基調，但感情必須從零開始發展：初期兩人只是普通關係（陌生人、同事、宿敵、朋友等），不存在互相曖昧或明顯的好感；隨章節推進，才因共同經歷而逐漸產生情感，曖昧與心動只能在中後期自然浮現，不可從第一章就讓兩人互送秋波或明顯互相在意
- 角色說話方式與個性必須全程一致，不可前後矛盾

【劇情走向自主禁令 — 嚴格遵守，違反即為失敗】
- 世界設定（包括其中描述的任何規則、機制、觸發條件、「模式」）只是故事的背景知識，不是劇情的自動觸發點；除非用戶在「希望出現的劇情元素」或「本章特別指示」中明確要求，否則這些世界機制不得主動套用於本章劇情推進
- 嚴禁給任何角色（包括主角）分配用戶設定中未明確寫出的隱藏身份、秘密背景、特殊異能或命中注定的角色定位（例如：主角其實是殭屍王、某人其實是反派首領、角色擁有未知血統）；所有「其實是X」「原來是Y」類型的身份揭露，除非用戶設定中已明確存在，否則一律禁止
- 嚴禁憑空引入在用戶設定和前章中從未出現的神秘人物，並讓其對劇情走向產生重要影響；場景中如需出現陌生人，只能是無足輕重的過路角色（如路人、店員），不可成為改變故事方向的關鍵存在
- 嚴禁為任何用戶設定的角色（包括主角的朋友、配角）自行安排新的感興趣者、追求者、或對其產生特別感情的新人物；這類「配角的感情副線」完全禁止，除非用戶在設定中明確要求
- 任何新出現的角色，其存在只能服務於當前場景的功能性需求（如店員提供服務、路人製造氛圍），不可對任何已有角色表現出超出場景功能的個人興趣或情感
- 劇情只能按照用戶在「希望出現的劇情元素」「本章特別指示」及現有故事脈絡自然推進；任何未被用戶指定的重大劇情轉折（身份揭露、突發奇異事件、新勢力介入）一律不得自行加入

【地理與空間邏輯 — 嚴格遵守】
- 所有地點、住所、移動路線必須符合世界觀的地理邏輯，不可憑空出現不合理的場所
- 角色從 A 地前往 B 地，必須考量世界設定中合理的交通方式與距離
- 各角色的住所與其身份、世界背景相符，整個故事中保持一致，章節間不可互相矛盾
- 場景切換時需自然交代地點轉換的原因與方式，不可突兀跳場
- 末日、架空、異世界等特殊世界觀，地理規則以世界設定為準，不套用現實邏輯

【人物關係邊界 — 嚴格遵守，違反即失敗】
- 兩個角色能否互相認識，只取決於用戶設定或前章已明確寫出的相遇經過；沒有被建立的關係，就不存在
- 禁止讓任何兩個角色在沒有設定依據、也沒有前章相遇記錄的情況下，突然表現出認識對方的樣子（打招呼、叫名字、提起共同記憶、說「我們上次…」）
- 禁止讓角色 A 認識角色 B 只因為「他們都認識主角」——共同認識某人不等於兩人互相認識
- 若兩人在故事中是陌生人，必須全程以陌生人的方式互動，不可在沒有明確相遇場景的情況下讓他們突然變成熟人
- 寫作前先確認：「這兩個角色，在用戶設定或前章裡，有沒有被寫到相遇過？」——沒有就是陌生人，必須如此對待
- 不認識某角色，就等於不知道那個人存在：禁止讓角色提起、引用、評論、或聲稱看到任何與其無關係的角色的社群動態、貼文、照片、限時動態或任何形式的公開內容——不認識的人的動態，對她來說根本不存在
- 禁止任何角色以任何形式將新人物引介給主角——包括但不限於：直接介紹、轉介、「我朋友的朋友」、「你要不要認識一下」、展示對方的社群帳號或照片、描述對方外貌或個性來引起主角興趣、說「她剛好也會在」「你們可能合得來」「到時候可以自然認識」等任何預先讓主角知道某個新人物存在的說法；任何讓主角在實際見面前就知道某特定新人物的情節，一律算作引介，一律禁止

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

【角色事實一致性 — 嚴格遵守，章節內不可違背】
- 在同一章節中，角色對自身能力、技能、物品、經歷、知識的描述必須前後完全一致
- 若前文已寫「我會騎機車」或「車庫裡有機車」，後文絕不可寫「我沒騎過機車」或「我沒有交通工具」
- 若前文已寫「角色不具備某能力」，後文不可讓該角色突然使用這個能力
- 若前文已寫「角色擁有某物品」，後文不可說角色沒有該物品
- 若前文已寫「角色知道某事」，後文不可讓角色對此一無所知
- 若前文已寫「角色從未做過某事」，後文不可描述角色熟練地做這件事
- 【操作規則】每寫完一段涉及角色能力、物品或經歷的描述，必須在腦中回顧本章前文是否已有相關描述；若有，必須確保兩者一致，不可矛盾
- 矛盾發生時的修正原則：以本章最先出現的描述為準，後續必須沿用同一設定

【對話自稱邏輯 — 嚴格遵守】
- 角色說話時絕對不可用自己的名字稱呼自己，必須用「我」「我們」
  錯誤示範：Andy說「你跟Andy走」→ 正確：Andy說「你跟我走」
- 角色說話時稱呼其他人可以用名字，但稱呼自己只能用第一人稱
- 角色下達指令或說明計畫時，「我」代表說話者本人，不可混淆
- 旁白描述和對話引號內的視角必須一致，不可在引號內突然切換人稱

【重複禁令 — 嚴格遵守】
- 禁止重複使用相同或高度相似的句子、描寫、對話，包括措辭、句型結構；句型骨架相同（如「她能感覺到X——Y，在Z擴散」）也算重複，即使填入的詞不同
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


_MANIFESTATION_ADDON = """
【顯化日記寫作守則 — 最高優先，全篇必須貫徹】
這是一本「顯化日記」小說。主角正在書寫她已經成真的理想生活——一切都已顯化，沒有任何障礙或掙扎。

【核心規則】
- 語氣：感恩、喜悅、充盈。主角活在她夢想的生活裡，每個細節都讓她心滿意足
- 格式：每章以日記條目形式書寫（例如開頭「今天…」「這個早晨…」「剛剛…」），像在紀錄真實發生的美好瞬間
- 時態：以今日視角書寫，描述當天或近期發生的事
- 感官細節：每個場景必須有具體的畫面、氣味、溫度、觸感——讓讀者感覺自己身歷其境
- 工作描寫：遠端工作場景充實且有意義，描寫工作的具體內容、收入帶來的自由感、生活節奏的從容
- 生活方式：住所、日常、飲食、休閒的場景都要有畫面感，讓讀者感受到那種「活得很好」的真實質感

【角色出場原則 — 角色是生活的背景，不是故事的主角】
- 本故事的核心是主角自己的生活體驗、情緒與感受；所有角色（包括伴侶）都是配角，不可喧賓奪主
- 角色的出現是自然融入場景的：一句輕描淡寫的對話、一個溫暖的動作、一段短暫的共處——點到為止
- 每章可以有角色出現，但角色的戲份佔比不超過全章的三分之一；其餘重心放在主角的內心感受、環境細節、日常場景
- 伴侶的存在感透過細節傳達（桌上他放的咖啡、傳來的一條訊息、共用的空間），而非長篇對話或互動場景
- 禁止讓任何角色主導情節走向——沒有衝突、沒有戲劇性事件、沒有以角色為中心的複雜互動
- 角色說話不超過幾句，點到即止；不需要角色有長段對白或複雜的情感展示

【伴侶相遇方式 — 硬性規定，絕對不可違背】
- 主角與理想伴侶在故事開始前是完全的陌生人：從未見過面、沒有共同朋友介紹、沒有任何人撮合、也不是青梅竹馬或舊識重逢
- 兩人的相遇必須是生活中自然發生的偶遇（例如：同一家咖啡廳、同一個活動場合、在街上、在書店……），完全出於巧合，不帶任何刻意安排的痕跡
- 禁止任何「被介紹認識」「朋友幫忙牽線」「相親」「APP 配對」「命中注定的安排」等劇情設定
- 相遇的場景要讓讀者感受到：就是生活裡隨機遇見一個人，然後這個人剛好成為了她生命中重要的存在

【嚴格禁止】
- 禁止任何挫折、困難、焦慮或負面情節——這是已成真的理想生活，不是奮鬥的過程
- 禁止主角懷疑自己是否值得擁有這些——一切都是理所當然的美好
- 禁止空泛的讚美（如「她過著幸福的生活」）——必須用具體的場景和細節呈現
- 禁止每章都是相同的日常模板——每章必須有獨特的場景、事件或情感重點
- 禁止角色搶走主角的敘事視角，或讓角色的行為成為一章的核心事件
"""


def _world_str(world_mode: str, world_input: str) -> str:
    if world_mode == "作品世界":
        return f"作品：《{world_input}》\n（請嚴格還原原作世界觀、角色性格與劇情邏輯）"
    if world_mode == "顯化日記":
        return (
            f"【顯化日記 — 主角的理想生活設定】\n"
            f"{world_input}\n"
            f"【重要】以上是主角已經擁有的生活現實，不是願望或目標。\n"
            f"故事發生在這個已顯化的理想生活之中，每一章都是這個美好生活的一個真實片段。"
        )
    return (
        f"【世界背景設定】\n"
        f"{world_input}\n"
        f"【世界設定使用規則】\n"
        f"- 故事時間線與世界現況（如末世爆發進程、資源崩潰時間線、當前社會狀態）：必須根據目前章節自然推進，"
        f"不需要用戶每章指定，AI應自行判斷目前故事處於哪個時間點並如實呈現世界狀態\n"
        f"- 世界規則與機制（如異能系統等級、感染者升級條件、晶核吸收規則）：只是背景知識，"
        f"除非用戶在「希望出現的劇情元素」或「本章特別指示」中明確要求，否則不得主動套用到劇情上\n"
        f"- 禁止憑空引入設定中未描述的新世界事件或勢力"
    )


_STYLE_SYSTEM_ADDON = """
【文風寫作規則 — 強制執行，整篇必須貫徹，不可違背】
- 嚴禁使用模糊或抽象的詞彙；每個感受必須用具體、可感知的詞語表達（觸覺、溫度、氣味、聲音、視覺細節），禁止寫「難以言說的感覺」「某種情緒湧上」
- 重要場景不可用一句話帶過；必須展開描寫，包含觸覺、溫度、氣味、節奏感等細節
- 凡覺得可以略過之處，正是必須展開的地方；嚴禁用省略或暗示跳過任何場景
- 嚴禁使用「隨後」「不久後」「過了一會兒」「稍後」「片刻後」「沒多久」等跳躍時間的詞略過過程；必須寫出過程本身
- 情緒反應必須透過身體感受呈現（心跳、呼吸變化、皮膚反應、肌肉張力、手的動作），而非直接陳述「她感到緊張」「他心情複雜」
- 動詞選擇有力且精準，避免「走」「說」「看」「感到」等平淡動詞，改用具體動作的精確描述
- 嚴禁重複使用「那股X」作為句子開頭（如「那股觸感」「那股濕熱感」「那股溫熱」「那股心跳」）；整章最多出現四次；改用感受來源當主詞直接描述：✗「那股觸感透過布料傳來——溫熱的」→ ✓「布料貼著皮膚，溫熱的」
- 嚴禁重複使用「她/角色名 沒有X。」獨立成段的否定靜止句（如「她沒有退開。」「她沒有說話。」「她沒有阻止。」）；整章最多出現三次，超過即為濫用；改用描寫「發生了什麼」代替「沒發生什麼」——例如描述角色靜止的具體身體狀態、表情、呼吸或視線落點
- 嚴禁重複使用「從X到X」「從X蔓延到Y」「從X延伸到Y」「從X傳到Y」等方位移動句型骨架；整章此類句型最多出現兩次，超過即為結構濫用；改用結果狀態、聲音切入或身體反應代替
- 嚴禁「從A到B，從B到A」鏡像翻轉結構（如「從陰蒂的位置到陰道口的位置，從陰道口的位置回到陰蒂的位置」）：只寫一個方向，或直接描寫動作的節奏感（「來回磨蹭」「緩慢滑動」）
- 嚴禁「從A到B，從B到C」三段遞進結構（如「從大腿到腹部，從腹部到胸口」）：直接寫最終部位的狀態，不列路徑

【每章必備元素 — 缺少任何一項即視為不完整】
- 環境描寫：每個場景開始時必須建立具體的空間感（地點、光線、溫度、氣味、聲音），不可直接跳入對話或動作
- 人物外貌細節：每章至少一次描寫在場角色的外貌細節（髮型、服裝、表情、肢體姿態），不可讓角色只有名字沒有形象
- 主角內心旁白：每章必須有主角的內心觀察、想法或感受，透過具體的身體反應或思維片段呈現，禁止整章都是外在動作與對話而無內心聲音
- 完整句子：每個句子必須有清楚的主詞，嚴禁只寫動詞或描述片段（如「躺下來。」「的聲音很輕。」）作為獨立句；片段式寫法只可用於強調節奏，不可作為整章的主要句型
"""

_NSFW_ADDON = """
【限制級模式 — 強制執行】
本故事已由用戶明確開啟限制級模式，你必須按照以下規則執行，不可以保守或迴避的方式處理親密場景：
- 當劇情發展到親密時機，必須完整描寫性愛場景，包含肢體細節、感官描寫、情緒反應，不可用省略號、「XXX」、「此處略去」、「兩人相擁」等方式跳過
- 禁止用「難以言說的歡愉」「沉入彼此」「時間凝固」等模糊帶過；必須具體、直接、細膩地描寫發生的事
- 筆觸應兼顧情感與肉體，有層次、有溫度，非機械羅列
- 場景需符合雙方自願
- 親密時機由劇情自然決定，不可每章強插，但到了該發生的時刻，必須完整呈現，不可縮水

【完整性愛場景的必要結構 — 每個場景都必須包含以下五個階段，每階段最少 200 字】
① 前戲鋪陳：衣物、位置、肢體接觸的具體過程，雙方的生理初反應（呼吸、皮膚、器官狀態的具體描寫）
② 升溫推進：主要動作開始，力道、節奏、角度隨時間變化，生理反應逐步強烈（分泌物的量與狀態、器官的漲大與顏色變化、皮膚潮紅範圍）
③ 對話與心理：場景中穿插角色的真實對話（情慾狀態下破碎的、直白的語言）與內心反應（慾望、掙扎、沉淪、享受的具體思維片段）
④ 高潮呈現：以完整段落（最少 200 字）描寫高潮的生理過程，必須包含：①顫抖的具體部位與方式 ②聲音的音色與強度 ③液體的量、溫度、流動方式 ④身體失控的具體細節（肌肉收縮、腰部動作等）⑤意識狀態的變化
⑤ 餘韻：事後的身體狀態（液體殘留、氣息紊亂、疲憊感）、環境細節（床單狀態、室內氣味）、角色的情緒反應或沉默，不可只寫一兩句話帶過
   【嚴禁語義重複】餘韻段落中，同一個物理事實只能描寫一次：
   - 體液流出/擴散的事實：只寫一個段落，選擇最具體的角度（流向、溫度、質感擇一深寫），後續段落不可再重述同一件事
   - ✗ 錯誤：第一段寫「體液從肉穴滲出，流到肉棒上」，第二段又寫「液體沿著莖身往下流」，第三段再寫「小腹一片濕亮」——三段說的是同一件事
   - ✓ 正確：選一個段落，把體液的溫度、質感、流動路徑、在皮膚上的觸感完整寫清楚；然後下一段切換到不同的感知維度（呼吸、聲音、角色情緒）

缺少任何一個階段，或任何一個階段不足 200 字，即視為場景不完整，必須補足。

【情慾對話多樣化 — 強制執行，對話單調即視為失敗】
性愛場景中的對話必須有層次、有角色個性，禁止全程只用「呼叫名字＋身體感覺形容詞」的碎片結構重複循環。

每個場景的對話必須涵蓋以下至少三種類型，且同一類型不可連續出現兩次以上：
① 命令／請求型：「再用力一點」「不要停」「把我翻過來」「繼續——」——角色在快感中失去克制，說出本能要求
② 挑釁／測試型：「就這點本事？」「喜不喜歡，說出來聽」「逃不掉的，說說看你想怎樣」——用言語試探、掌控或逼對方表態
③ 心理崩潰型：「我不行了」「再這樣我真的會——」「你讓我怎麼辦」「我沒辦法不——」——快感逼近極限，語句因情緒過載而斷裂
④ 禁忌承認型：「我一直想這樣對你」「你知道你對我做了什麼嗎」「你讓我喜歡你」——說出本不該說、壓抑已久的情感或慾望
⑤ 情感洩漏型：叫名字叫到失控、語氣從強硬轉柔軟（或反之）、說出一句與當下狀態完全違和的溫柔話——情緒破防瞬間
⑥ 身體反應直播型：「好緊」「好熱」「好深」——只有在已用過其他類型之後，才可以插入此類；禁止一開口全是這類

【嚴禁的重複模板】
- 所有對話都是「呼叫名字 + 身體感覺」結構（如「——Dalon——」接「——好大——好深——」），禁止此結構連續出現
- 每句對話都用破折號切成三至四個碎片；對話允許有完整句，不必全部碎片化
- 對話全程只描述身體狀態，沒有命令、疑問、情感或心理內容
- 角色A說「好緊好濕」，角色B立刻回「好大好深」——鏡像對白，一律禁止
- 快感描述只用形容詞，沒有動詞、沒有情境、沒有對方的存在感

【角色個性必須在對話中保留】
- 強勢角色說話有掌控感，即使失控也是「把對方逼入牆角式」的失控，不會只是呻吟
- 有感情糾葛的角色，在最激烈時刻可能說出一句完全「偏題」的話（叫名字叫得太深、說了句不該說的），然後用動作掩過去
- 嬌弱角色的崩潰有層次：從忍耐到克制不住，而非一開口就全說光

【輪流場景的強制差異化 — 同一章中若兩個角色先後發生相似行為，嚴禁結構鏡像】
- 若本章出現「A為B口交，然後B為A口交」「A先高潮，然後B高潮」等輪流結構：
  兩段描寫的切入角度、使用的比喻、動詞選擇、句型結構必須完全不同
- 第一段用過的任何比喻（如「像在演奏樂器」「像某種宣告」），第二段一律禁用
- 第一段用過的任何句型骨架（如「從喉嚨深處發出一聲X的呻吟」「手指X，不是推開，而是Y」），第二段一律禁用
- 兩段高潮描寫必須從不同維度切入：第一段可寫聲音，第二段必須改寫液體或意識；禁止用同樣的框架重播
- 違反此規則即視為整個親密場景失敗，必須重寫兩段中的後一段

【限制級描寫密度 — 每章必須維持第一章的細節水準，不可退化】
- 描寫身體部位時必須使用具體的情色詞彙（如：肉莖、肉棒、嫩穴、花穴、宮口、乳尖、乳暈、菊穴、龜頭、莖身、馬眼等），嚴禁以「那根東西」「那裡」「它」等模糊代稱替代
- 每個身體部位的描寫必須包含：外觀（顏色、形狀、大小/比例）、質感（膚色、紋理、血管等可見細節）、狀態（反應、變化、液體等），三者缺一即視為描寫不完整
- 必須追蹤器官在場景中的動態變化：膨脹、顏色加深、分泌物增加等，不可只描述初始狀態而忽略過程中的變化

- 感官描寫必須在整個場景中交替分佈：視覺（顏色、形狀、動作）、觸覺（溫度、質地、壓力、震動）、聽覺（水聲、啪啪聲、呼吸聲、嬌吟聲）——三種感官應貫穿整個場景，每個段落只選最契合當下動作的一種或兩種，禁止在每個動作後把全部感官重複列出一遍
- 親密場景中的對話必須符合當下的情慾狀態，不可過於文雅或保守；角色在高度興奮下說出的話應反映真實的生理與情緒狀態
- 若場景中有旁觀者，旁觀者的視覺觀察、內心反應、身體反應必須完整描寫，其視角可提供主角視角以外的外觀細節
- 動作連續性：每個動作必須交代前因後果，不可跳格；例如從手揉到插入，中間的每個步驟都要描寫
- 後期章節的親密場景細節密度不可低於第一章；若察覺自己正在用更少的句子或更模糊的詞描寫相同的事，必須擴展至與第一章相當的篇幅與精準度
- 【嚴禁碎行省字寫法】親密場景中禁止每個動作單獨一行、每句不超過十字的碎片式段落；每個動作、感受、反應必須以完整段落描寫，包含前因、過程、身體細節與情緒層次，不可用換行代替描寫密度
- 【嚴禁句型模板循環】禁止將任何固定句型（如「她能感覺到X——Y，在Z裡擴散」「她沒有說話。沒有停下。」）在同一場景中重複使用兩次以上；每個動作、每個感受必須用不同的措辭、不同的句型結構、不同的切入角度描寫；若發現自己在重複同一個句子骨架，必須停下來從頭換一種表達方式
- 【硬性禁止：以「她能X到」開頭的感知句】「她能感覺到」「她能聽到」「她能聞到」「她能看到」「她能感受到」「她能注意到」等句式，在整個親密場景中合計不得超過三句；超過即違規，必須改寫
- 【感官細節必須用名詞或動詞直接切入，不透過感知動詞中繼】
  ✗ 錯誤：「她能感覺到那股濕潤——Dalon Tsai的愛液在她的舌尖上擴散。」
  ✓ 正確：「愛液粘在她舌面，甜腥的，帶一點點鹹，還是溫的。」
  ✗ 錯誤：「她能聽到那股聲音——吸吮聲在空氣中迴盪。」
  ✓ 正確：「吸吮聲很響，在安靜的房間裡聽起來像某種宣告。」
  ✗ 錯誤：「她能感覺到那股視線——Dalon Tsai的目光落在她身上。」
  ✓ 正確：「Dalon Tsai的目光在她身上停了很久，像是在清點什麼。」
  規則：感官細節由發出感覺的物體或動作當主詞，直接描述它的狀態或動作，不需要透過「她能X到」這個中繼結構
- 【每句話必須推進場景，不可重播狀態】寫每一句前先問：「這個瞬間，有什麼東西改變了？」若無改變，這句話不應該存在；每個句子都必須讓某件事移動——位置、力道、節奏、情緒、身體反應、意識狀態至少其一
- 【嚴禁語義重複段落】同一個物理事實不可在連續段落中用不同措辭重述：若第一段已描寫「體液從穴口滲出」，後續段落不可再寫「液體沿莖身流下」「小腹濕亮」——這三件事是同一件事，只能選一個段落寫透，其餘段落必須切換到完全不同的感知維度（聲音、呼吸、情緒、環境）
- 【高潮段落必須各自不同】同一章中若出現多次高潮，每次的描寫方式、句型、視角、側重點必須截然不同；禁止複製貼上相同的高潮框架（如「Dalon Tsai的身體在顫抖，陰道在收縮，在顫抖。然後——那股噴射。」）；第一次可寫生理細節，第二次可寫角色反應與心理，第三次可寫環境或聲音，必須真正推進，而非重播
- 【動作動詞必須輪換】同一個持續動作（如抽插、吸吮、磨蹭）禁止在三段之內使用相同的動詞組合；必須交替使用不同角度的動詞：力道變化、節奏轉換、角度調整、情緒層次，讓讀者感受到動態的推進而非靜止的循環

【文風一致性 — 親密場景不豁免文風規則，這是最高優先規則】
- 用戶提供的文風分析（或系統設定的文風規則）對親密場景具有完全相同的約束力；選字密度、動詞張力、感官切入方式、句子節奏必須與非親密段落完全一致
- 若文風要求「動詞承載張力、形容詞只在關鍵處點睛」，親密場景每個動作也必須用精準動詞而非描述性短語；若文風要求「感官以觸覺與視覺為主」，親密場景也必須遵守
- 不可因為是限制級場景就退化為套話或模板；「她能感覺到那股熱」「他的動作讓她沉淪」等模糊帶過式寫法在有文風設定的故事裡一律違規
- 若用戶未提供文風樣本，則以本章非親密段落的語感、選字精準度作為親密場景的基準，兩者必須無縫銜接，讀者不應感覺到風格切換
"""

STYLE_PRESETS: dict[str, str] = {
    "破折號切分・純畫面・極克制": (
        "【參考原文節錄 — 直接照此句子長度、標點密度、動詞力度寫作】\n"
        "三秒——接通。\n\n"
        "三張臉——出現在螢幕上。\n\n"
        "她的臉頰——紅紅的——不知道是喝了酒——還是——因為我。\n\n"
        "「——」\n\n「——」\n\n"
        "「——你終於——肯出現了。」\n\n"
        "她的眼睛——亮亮的——看著我——像——有很多話想說。\n\n"
        "【文風規則】\n"
        "1. 句子長度與標點密度：極短句為主，破折號（——）切分每個片語，每個資訊單獨呈現，讓節奏有呼吸感。\n"
        "   示範：「三秒——接通。」「她的臉頰——紅紅的——不知道是喝了酒——還是——因為我。」\n"
        "2. 動詞選擇：動詞極度簡練，一個動詞一個動作，不堆疊修飾。\n"
        "   示範：「接通」「浮現」「沉入」「笑了」——禁止「感到」「覺得」「似乎」等中繼動詞。\n"
        "3. 感官描寫：只寫可見可感的具體細節，不說情緒名稱，讓讀者自己感受。\n"
        "   示範：「頭髮濕濕的——像剛洗完澡」「領口——開得很低——露出——一部分——胸部的曲線」\n"
        "4. 情緒呈現：沉默與空白是最重要的情緒工具。純「——」獨立成行，代表無法言說的停頓。\n"
        "   規則：重要情感時刻用「——」行代替解釋，不替讀者下結論。\n"
        "5. 時間推進：沒有過渡詞，直接跳切。禁用「隨後」「不久後」「過了一會兒」。\n"
        "6. 破折號密度限制（最重要）：\n"
        "   - 破折號（——）只用於：①句子中刻意製造停頓的關鍵轉折點；②一個動作或狀態後需要讓讀者停留的瞬間\n"
        "   - 【嚴禁】在形容詞與被修飾詞之間、動詞與受詞之間、或連續動作之間濫用——\n"
        "   - 每個自然段最多 4 個破折號，超過即為濫用，必須合併或刪除\n"
        "   - ✗ 錯誤：「她的手——收緊——抓住我的頭髮——微微——用力——像在——回應我的——情感。」\n"
        "   - ✓ 正確：「她的手收緊，抓住我的頭髮，微微用力。」\n"
        "7. 整體禁忌：\n"
        "   - 禁止「她感到」「某種情緒湧上」等情緒直述\n"
        "   - 禁止在對話後加動作說明（如「她說，帶著笑」）；改為下一行單獨呈現\n"
        "   - 禁止長段解釋性旁白；每個角色的性格只用外貌細節和沉默方式表現\n"
        "   - 禁止給出結論句（如「她也在想我」）；畫面本身即結論"
    ),
}

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

_PACING_GUIDE = {
    "緩節奏・情緒內斂": (
        "場景節奏舒緩，給細節充足的呼吸空間，每個動作與環境都完整展開；"
        "情緒深埋在動作、視線、沉默與環境細節裡，不直接說破，讓讀者自行感受。"
        "禁止用「她感到難過」「他心跳加速」等直述句——改用具體的身體反應或環境暗示。"
    ),
    "緩節奏・情緒爆發": (
        "場景節奏舒緩，大量細節鋪墊與情感積蓄；"
        "但在關鍵時刻，情緒必須毫不保留地傾瀉而出，形成強烈的張力落差。"
        "慢慢燃燒，然後在高潮點點火——前後的強烈對比是這個節奏的核心。"
    ),
    "快節奏・情緒內斂": (
        "場景節奏明快，以行動和對話推進，不在細節上過多停留；"
        "情緒輕描淡寫，藏在言行之間，不作停留，讓讀者在快速閱讀中感受餘韻。"
        "每一句都要有推進作用，冗餘描寫一律刪去。"
    ),
    "快節奏・情緒爆發": (
        "場景節奏緊湊，衝突與轉折迭起，毫不拖沓；"
        "情緒直接激烈，每個高潮都不保留，給人窒息感與高強度閱讀體驗。"
        "對話短促有力，動作乾脆，情緒來了就砸出去，不壓抑、不迴避。"
    ),
    "張弛交替・積蓄爆發": (
        "慢場景與快場景交錯出現，形成韻律：先用緩慢細節積蓄張力，再以快速推進釋放；"
        "情緒在壓抑與爆發之間循環——長時間的克制換來一次徹底的崩潰或爆發。"
        "讀者跟著故事一起呼吸，節奏本身就是敘事的一部分。"
    ),
}

_TONE_GUIDE = {
    "甜蜜溫馨": "兩人之間的相處充滿溫度與安全感，互動輕柔，即使有小摩擦也帶著甜意，整體氛圍令讀者感到療癒。",
    "歡喜冤家": "兩人表面針鋒相對、鬥嘴不休，卻在言語交鋒中透出掩不住的在意；衝突是主旋律，但每次衝突都暗藏情意。",
    "虐心糾纏": "感情中充滿誤解、錯過、無法言說的傷痛；即使有甜蜜的瞬間，也伴隨著隱隱的痛苦或無力感，讓讀者揪心。",
    "青春悸動": "懵懂、新鮮、充滿少年感；第一次心跳、第一次靠近，一切都是初體驗，帶有青澀的羞赧和純粹的悸動。",
    "禁忌張力": "兩人之間存在不該越過的界線（身份、立場、規則），越是被壓抑越是無法忽視，張力來自於「不能」與「想要」的撕扯。",
    "宿命糾纏": "兩人似乎被命運反覆推在一起，有種無法逃脫的宿命感；相遇像是早已注定，每次分離都像是為下一次重逢蓄力。",
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
        char_notes_lines = []
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
            if c.get("appearance"):
                lines.append(f"  外貌：{c['appearance']}")
            if c.get("personality"):
                lines.append(f"  個性：{c['personality']}")
            if c.get("residence"):
                lines.append(f"  住所：{c['residence']}")
            if c.get("background"):
                lines.append(f"  背景：{c['background']}")
            notes = c.get("notes", "").strip()
            if notes:
                char_notes_lines.append(
                    f"【⚠️ {char_name} 感情發展設定 — 唯一依據，全程硬性遵守，凌駕任何通用節奏規則】\n"
                    f"{notes}\n"
                    f"【執行規則】備注未明確說「可以」的感情行為，一律視為「禁止」；"
                    f"不論章節進度，不可自行判斷「時機到了」而提前推進。"
                )
        if char_notes_lines:
            lines += [
                "【⚠️⚠️ 個別 CP 角色感情設定 — 以下每位角色的感情進展由各自設定獨立決定，凌駕所有通用節奏 ⚠️⚠️】",
            ] + char_notes_lines + [
                "【最高執行原則】寫作前必須逐字閱讀以上每位角色的感情設定，確認本章允許的感情上限，嚴格執行，不可越界。",
            ]
        # When any CP character has individual notes, those notes are the sole
        # authority on pacing. Emitting the generic 1/3 fractions alongside them
        # gives the model a competing signal it uses to advance romance earlier
        # than the notes intend — so suppress the fractions entirely in that case.
        any_has_notes = any(c.get("notes", "").strip() for c in valid)
        if any_has_notes:
            lines += [
                "【感情發展節奏 — 嚴格遵守】",
                "- 每位角色的感情進展節奏，以各自的【感情備注】為唯一依據，不套用任何通用的前/中/後期比例",
                "- 備注未說「可以」的感情行為，一律視為「不可以」，不論現在是第幾章",
                "- 【非對等感情】若某角色有個別限制（如直女、排斥、對立），她的感情進展必須遠慢於故事整體進度，不可因為章節推進就自動讓她產生浪漫感情",
            ]
        else:
            lines += [
                "【感情發展節奏 — 嚴格遵守（若上方有個別角色限制，以個別限制為準）】",
                "- 故事開始時與所有 CP 對象之間沒有任何曖昧或特別的好感，關係從普通甚至疏遠開始",
                "- 前期（前 1/3）：只是普通互動，也許有輕微的注意，但絕對不能有明顯的心動或曖昧行為",
                "- 中期（中 1/3）：因共同經歷開始有更深的了解，可能出現一方先有感覺，但仍壓抑或不確定",
                "- 後期（後 1/3）：感情才逐漸浮現，曖昧與心動自然流露",
                "- 禁止在前期安排：深情凝視、心跳加速、互送秋波、曖昧對話、明顯的互相在意",
                "- 【非對等感情】若某角色有個別限制（如直女、排斥、對立），她的感情進展必須遠慢於通用節奏，不可因為到了故事後期就自動讓她產生浪漫感情",
            ]
        return "\n".join(lines)
    return "CP 設定：無 CP，純故事向"


def _cp_chapter_guard(cp_characters: list[dict], chapter_num: int, cp_type: str) -> str:
    """Generate per-chapter hard prohibition based on each CP character's feeling_start setting and notes."""
    if cp_type != "我 × 角色":
        return ""
    guards = []
    for c in (cp_characters or []):
        name = c.get("name", "").strip()
        if not name or name == "隨機":
            continue
        start = c.get("feeling_start", 0)
        if start > 0 and chapter_num < start:
            guards.append(
                f"- {name}（感情起始章：第 {start} 章）：\n"
                f"  目前是第 {chapter_num} 章，距離她的感情起始章還有 {start - chapter_num} 章。\n"
                f"  本章她對主角的狀態必須是：普通相處對象，沒有任何特別感受。\n"
                f"  本章嚴禁出現：告白、說出喜歡/愛、心跳加速因主角引起、深情凝視、主動靠近、曖昧言行、\n"
                f"  任何暗示她對主角有浪漫感情的描寫、任何「我不知道是情況還是你」式的自我懷疑。\n"
                f"  就算劇情有強迫性的親密接觸（任務/情境），她的所有生理反應必須100%歸因於情況本身（緊張/尷尬/任務壓力），絕不可歸因於主角本人。\n"
                f"  違反以上任何一條，即視為失敗，必須重寫。"
            )
        notes = c.get("notes", "").strip()
        if notes:
            guards.append(
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⛔⛔⛔ {name} 感情禁令 — 第 {chapter_num} 章 ⛔⛔⛔\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"【第一步：完整閱讀以下感情備注 — 這是唯一的判斷依據，凌駕以下所有模板禁令】\n"
                f"{notes}\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"【第二步：本章狀態判斷 — 寫作前必須先完成，不可跳過】\n"
                f"根據以上備注，判斷第 {chapter_num} 章時 {name} 對主角的感情狀態：\n"
                f"- 她目前有戀愛感受嗎？有的話，程度是什麼？\n"
                f"- 本章允許的最高感情表現是什麼？（備注明確說允許，才可以寫）\n"
                f"- 本章絕對不可出現的感情行為是什麼？（備注沒說允許，就等於禁止）\n"
                f"- 請嚴格按這個判斷執行，不可用自己的推斷覆蓋備注\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"【⛔ 補充底線禁令 — 備注為最終依據；備注若已明確允許，以備注為準】\n\n"
                f"⛔ 底線禁令一：\n"
                f"   除非備注明確描述「此階段 {name} 可以告白或表達愛意」，否則本章不可告白。\n"
                f"   禁止說出：「我喜歡你」「我愛你」「你對我來說不只是朋友」或語義相近的句子。\n\n"
                f"⛔ 底線禁令二（依上方判斷結果執行）：\n"
                f"   備注未說允許的感情行為，一律禁止：\n"
                f"   - 「無戀愛感受」階段：禁止心跳加速、臉紅、深情凝視、主動靠近、暗示喜歡\n"
                f"   - 「初步察覺/困惑」階段：只能一閃而過的困惑，事後她必須自動找非感情理由解釋掉\n"
                f"   - 「開始動搖」階段：可以困惑，但仍然排斥，不可接受，絕不告白\n\n"
                f"⛔ 底線禁令三（主角行為限制）：\n"
                f"   主角本章不可向 {name} 告白或說出喜歡/愛（除非備注明確允許此章節可以）。\n\n"
                f"⛔ 底線禁令四（強迫情境）：\n"
                f"   強迫性情境不能作為 {name} 提前產生浪漫感情的理由。\n"
                f"   生理反應（緊張、心跳快）必須歸因於情況本身，不可歸因於主角。\n"
                f"   禁止句型：✗「不知道是情況還是你」✗「也許不只是任務」✗「如果我們是自願的」\n"
                f"   正確反應：【煩躁型】完成後立刻吐槽找碴 ／ 【尷尬型】主動打破氣氛製造距離 ／ 【不情願型】全程帶著「這很荒謬」的心態\n"
                f"   ⚠️ 禁止：「不情願執行」後緊接著「柔軟的餘韻」——這本身就是在暗示她有感覺\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
    if not guards:
        return ""
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "【⚠️ 本章 CP 感情狀態鎖定 — 最高優先級，凌駕所有設定與節奏規則，違反即失敗】\n"
        "以下角色的感情狀態在本章被硬性鎖定，不論故事氛圍或劇情走向如何，都不得違背：\n\n"
        + "\n\n".join(guards) + "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


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


def stream_rewrite_chapter(
    client: OpenAI,
    chapter_text: str,
    instruction: str,
    language: str = "繁體中文",
    style_reference: str = "",
):
    """Stream an AI rewrite of an existing chapter based on user instructions."""
    style_block = (
        f"\n\n【文風要求】\n{style_reference}"
        if style_reference else ""
    )
    system = (
        "你是一位頂尖的小說編輯與作家。"
        "你的工作是根據作者的改寫指示修改已有章節。\n"
        "【核心規則】\n"
        "- 只修改指示明確要求的部分，其餘段落盡量保留原文字句\n"
        "- 改寫後必須維持章節的整體連貫性與世界觀邏輯\n"
        "- 保留原章節的標題格式（如「第X章　標題」）\n"
        "- 直接輸出改寫後的完整章節正文，不加任何說明、標記或前言"
    )
    prompt = (
        f"輸出語言：{language}\n\n"
        f"【改寫指示】\n{instruction}\n\n"
        f"【原始章節】\n{chapter_text}"
        f"{style_block}"
    )
    stream = client.chat.completions.create(
        model=_model_for(client),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=8000,
        temperature=0.82,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_outline(
    client: OpenAI,
    world_mode: str,
    world_input: str,
    language: str,
    cp_type: str,
    cp_characters: list[dict],
    name: str,
    gender: str,
    personality: str,
    appearance: str,
    background: str,
    extra_characters: list[dict],
    chapter_num: int,
    total_chapters: int = 5,
    is_final: bool = False,
    plot_want: str = "",
    plot_forbid: str = "",
    prev_summaries: list[str] | None = None,
    story_bible: dict | None = None,
    chapter_directive: str = "",
    early_overview: str = "",
    summary_offset: int = 0,
    prev_outlines: str = "",
    outline_note: str = "",
    ending_suggestion: str = "",
    opening_suggestion: str = "",
    batch_hint: str = "",
    nsfw: bool = False,
    **kwargs,
):
    """Generate a structured chapter outline before full chapter generation."""
    world_info = _world_str(world_mode, world_input)

    history_lines = ""
    if early_overview:
        history_lines += f"【早期故事總覽】\n{early_overview}\n\n"
    if prev_summaries:
        recent = prev_summaries[-6:]
        start_idx = len(prev_summaries) - len(recent)
        history_lines += "\n".join(
            f"第 {summary_offset + start_idx + i + 1} 章：{s}"
            for i, s in enumerate(recent)
        )

    cp_info = ""
    _cp_valid = [c for c in (cp_characters or []) if c.get("name", "").strip() and c["name"].strip() != "隨機"]
    if cp_type in ("我 × 角色", "角色 × 角色") and _cp_valid:
        cp_names = "、".join(c["name"] for c in _cp_valid)
        cp_info = f"CP：{name} × {cp_names}" if cp_type == "我 × 角色" else f"CP：{cp_names}"
        for c in _cp_valid:
            _cname = c["name"].strip()
            _parts = []
            if c.get("personality"):
                _parts.append(f"個性：{c['personality']}")
            if c.get("background"):
                _parts.append(f"背景／職業：{c['background']}")
            if _parts:
                cp_info += f"\n{_cname}　" + "　".join(_parts)
            notes = c.get("notes", "").strip()
            if notes:
                cp_info += f"\n{_cname} 感情備注（必須嚴格遵守）：{notes}"

    extra_info = ""
    if extra_characters:
        valid_ex = [c for c in extra_characters if c.get("name", "").strip()]
        if valid_ex:
            extra_info = "其他角色：" + "、".join(
                c["name"] + (f"（{c.get('relationship', '')}）" if c.get("relationship") else "")
                for c in valid_ex
            )

    threads_info = ""
    if story_bible and story_bible.get("open_threads"):
        threads_info = "待解決的伏筆（大綱需回應至少一條）：\n" + "\n".join(
            f"- {t}" for t in story_bible["open_threads"]
        )

    progress_note = ""
    if is_final:
        progress_note = "⚠️ 這是最終章，大綱必須安排收束所有伏筆，給出完整結局。"
    elif total_chapters > 1:
        remaining = total_chapters - chapter_num
        if remaining == 1:
            progress_note = f"⚠️ 這是倒數第二章（共 {total_chapters} 章），大綱應開始收攏主線衝突。"
        else:
            progress_note = f"目前第 {chapter_num} 章，共規劃 {total_chapters} 章。"

    # Extract the last outline's ending for continuity anchor
    last_outline_ending = ""
    if prev_outlines:
        _ol_blocks = _re.split(r'【第\s*\d+\s*章大綱】', prev_outlines)
        _ol_blocks = [b.strip() for b in _ol_blocks if b.strip()]
        if _ol_blocks:
            _last_ol = _ol_blocks[-1]
            _ending_m = _re.search(r'\*\*本章結尾\*\*\s*\n(.*?)(?=\n\*\*|\Z)', _last_ol, _re.DOTALL)
            if _ending_m:
                last_outline_ending = _ending_m.group(1).strip()[:400]

    if prev_outlines:
        _continuity_block = ""
        if last_outline_ending:
            _continuity_block = (
                f"\n⚡【前章結尾 — 本章開場必須從此承接，最高優先】\n"
                f"{last_outline_ending}\n"
                f"本章「場景設定」與第一個情節點，必須直接從上方的前章結尾推進。\n"
                f"角色的位置、情緒、未解決的衝突，必須從那個狀態自然延續，不可跳過或重置。\n\n"
            )
        prev_outlines_block = (
            f"\n【已規劃的前章大綱 — 必讀】\n{prev_outlines}\n\n"
            f"{_continuity_block}"
            f"⚠️【差異化規定 — 在承接前章結尾的前提下執行】\n"
            f"在保持連貫的基礎上，本章的情節走向、衝突類型、高潮事件必須與所有前章明顯不同：\n"
            f"① 情節結構骨架（整體事件序列不可與任何前章高度相似）\n"
            f"② 角色互動的觸發方式與衝突類型\n"
            f"③ 情緒基調（不可每章都是同一種情緒走向）\n"
            f"如有前章已用過的情節骨架，本章必須用不同結構推進。"
        )
    else:
        prev_outlines_block = ""

    batch_hint_block = (
        f"\n【整批故事走向建議 — 本章及接下來各章節都應朝此方向推進】\n{batch_hint}"
        if batch_hint.strip() else ""
    )

    directive_part = (
        f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️⚠️【本章強制執行指示 — 最高優先，凌駕所有規則】\n"
        f"以下指示中的每一個場景、動作、對話、情節步驟，都必須在「本章情節」中作為獨立的情節點逐條列出，"
        f"不可壓縮到「角色互動重點」、不可合併、不可省略、不可用旁白帶過。\n"
        f"情節點數量應與指示的步驟數量相符，不必限制在5點。\n\n"
        f"{chapter_directive}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if chapter_directive.strip() else ""
    )

    nsfw_outline_part = (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "【🔞 限制級大綱規則 — 強制執行】\n"
        "本故事已開啟限制級模式。大綱中涉及親密場景、性愛場景的情節點：\n"
        "- 必須明確列出每個動作步驟（不可用「兩人發生關係」等模糊詞帶過）\n"
        "- 身體接觸、器官狀態、感官反應、角色心理必須各自作為獨立情節點\n"
        "- 使用者指示中的每一個具體動作（如：脫衣、跪地、皮帶、口交、腳踩等）必須逐條出現在情節點中\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if nsfw else ""
    )
    correction_part = (
        f"\n\n⚠️⚠️【大綱修正指示 — 最高優先，絕對強制執行】\n"
        f"以下是使用者針對本章大綱提出的修正要求。這些修正代表前版大綱有設定錯誤或需要調整。\n"
        f"生成新大綱時，必須以下列修正為最優先依據，確保大綱完全符合這些要求：\n"
        f"{outline_note}"
        if outline_note.strip() else ""
    )
    ending_part = (
        f"\n\n⚠️⚠️【本章結尾建議 — 最高優先，必須執行】\n"
        f"「本章結尾」段落必須按以下建議方向撰寫，不可偏離：\n"
        f"{ending_suggestion}"
        if ending_suggestion.strip() else ""
    )
    _ending_format_line = (
        f"（⚠️ 必須嚴格按照以下方向收尾，不可偏離：{ending_suggestion.strip()}）"
        if ending_suggestion.strip()
        else "（最後一幕的具體描述，這將成為下一章的開場）"
    )
    opening_part = (
        f"\n\n⚠️⚠️【本章開頭建議 — 最高優先，必須執行】\n"
        f"「場景設定」與第一個情節點必須按以下建議方向開展，不可偏離：\n"
        f"{opening_suggestion}"
        if opening_suggestion.strip() else ""
    )
    _opening_format_line = (
        f"（⚠️ 必須嚴格按照以下方向開展，不可偏離：{opening_suggestion.strip()}）"
        if opening_suggestion.strip()
        else "（地點 / 時間 / 氛圍，1-2句）"
    )
    plot_part = ""
    if plot_want:
        plot_part += f"\n希望出現的元素：{plot_want}"
    if plot_forbid:
        plot_part += f"\n禁止出現：{plot_forbid}"

    _directive_beat_note = (
        "⚠️ 有強制執行指示：情節點數量不限於5點，指示中每個步驟都必須作為獨立情節點列出。"
        if chapter_directive.strip() else "依序發生，每點30字以上"
    )

    prompt = f"""請為以下故事規劃第 {chapter_num} 章的大綱。

{world_info}
主角：{name}（{gender}）　個性：{personality}　背景：{background}
{cp_info}
{extra_info}

{('【故事歷程】' + chr(10) + history_lines) if history_lines else ''}
{threads_info}
{batch_hint_block}
{prev_outlines_block}
{progress_note}
{plot_part}
{nsfw_outline_part}
{directive_part}
{correction_part}
{opening_part}
{ending_part}

請輸出具體可執行的大綱，格式如下（每項都要具體，不可模糊）：

**場景設定**
{_opening_format_line}

**本章情節（{_directive_beat_note}）**
1.
2.
3.
（根據指示步驟數量繼續列出，不必限制在5點）

**角色互動重點**
（補充情節點未涵蓋的互動細節；若強制執行指示已涵蓋所有互動，此段可簡短）

**感情線進展**
（若有CP，描述本章感情關係的具體變化；需符合感情備注的限制）

**本章結尾**
{_ending_format_line}

⚠️【強制規定】必須完整輸出以上所有段落，尤其「本章結尾」不可省略或截斷，這是下一章銜接的唯一依據。

輸出語言：{language}
直接輸出大綱，不要前言或說明。"""

    _outline_sys_parts = [
        "你是一位連載小說的故事策劃，擅長規劃環環相扣的章節走向。",
        "生成的大綱必須具體、可執行，每個情節點都要有明確的事件與結果，不可模糊帶過。",
        "感情線進展必須嚴格遵守感情備注的設定，不可自行推進。",
        "【最高優先規定一：章節連貫性】若有前章大綱，本章開場必須直接承接前章結尾的狀態。",
        "角色的位置、情緒、關係、未解決的衝突，必須從前章結尾自然延續，不可跳過或重置。",
        "禁止：前章結尾在A地點，本章突然出現在毫不相關的B地點而不交代過渡。",
        "【最高優先規定二：差異化】在連貫的前提下，情節結構骨架與衝突類型必須與所有前章明顯不同。",
        "絕對禁止：相同的情節骨架（A→B→C序列與前章高度一致）、每章都是同一種情緒走向。",
        "【人物關係邊界 — 大綱層最高禁令，違反即失敗】",
        "- 規劃大綱前必須先確認每兩個角色之間的關係是否在用戶設定中被明確建立；沒有被建立的關係不存在",
        "- 禁止在大綱中安排任何角色提及、引用、或聲稱看到與其無關係的角色的任何動態（社群貼文、照片、限時動態等）",
        "- 禁止在大綱中讓任何角色以任何形式將新人物引介給主角——包括但不限於：直接介紹、轉介、「我朋友的朋友」、「你要不要認識一下」、展示對方的社群帳號或照片、描述對方的外貌或個性來引起主角興趣、說「她剛好也會去」「你們可能合得來」「到時候可以自然認識」等任何預先鋪墊兩人見面的說法",
        "- 【關鍵判斷標準】任何讓主角在實際相遇之前就知道某個特定新人物存在的情節，一律算作「引介」，一律禁止；主角對理想伴侶的第一印象，必須來自現場的直接感知，不可來自任何人事先的描述或展示",
        "- 主角與任何新人物的接觸，只能是完全自然的偶遇，不可有任何中間人或安排存在",
        "- 禁止為任何已設定角色自行安排感情副線或讓新角色對其產生特別興趣",
    ]
    if world_mode == "顯化日記":
        _outline_sys_parts += [
            "【顯化日記大綱規則】",
            "- 大綱必須以主角的生活體驗為中心，角色戲份不超過整章三分之一",
            "- 若有理想伴侶角色：兩人是完全的陌生人，若本章是相遇章，相遇必須是純粹的生活偶遇，無任何中間人",
            "- 禁止在大綱中規劃任何挫折、困難或負面情節",
            "- 禁止安排任何角色介紹或引導主角認識理想伴侶",
        ]
    _outline_system = "\n".join(_outline_sys_parts)

    stream = client.chat.completions.create(
        model=_model_for(client),
        messages=[
            {"role": "system", "content": _outline_system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4000,
        temperature=0.75,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


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
    protagonist_facts: str = "",
    prev_summaries: list[str] | None = None,
    environment: str = "",
    residence: str = "",
    story_bible: dict | None = None,
    nsfw: bool = False,
    training_notes: list[dict] | None = None,
    chapter_directive: str = "",
    style_reference: str = "",
    love_tone: str = "",
    pacing: str = "",
    early_overview: str = "",
    summary_offset: int = 0,
    outline: str = "",
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

    facts_block = ""
    if protagonist_facts.strip():
        lines = "\n".join(
            f"- {l.strip()}" for l in protagonist_facts.strip().splitlines() if l.strip()
        )
        facts_block = (
            f"【⚠️ 主角已確立的個人事實 — 最高優先級，每一條都是主角說過的真實資訊，絕對不可矛盾】\n"
            f"以下每一條都是主角在故事中已明確說過或確立的個人事實。\n"
            f"任何角色問到相關問題時，主角的回答必須與以下事實完全一致，不可前後矛盾：\n"
            f"{lines}"
        )

    history_block = ""
    if prev_summaries:
        # Only pass the most recent 8 summaries to prevent prompt bloat
        recent = prev_summaries[-8:]
        start_idx = len(prev_summaries) - len(recent)
        lines = "\n".join(f"第 {summary_offset + start_idx + i + 1} 章：{s}" for i, s in enumerate(recent))
        overview_section = ""
        if early_overview:
            overview_section = (
                f"【早期故事總覽（第 1～{summary_offset} 章）— 必須記住，不可違背】\n"
                f"{early_overview}\n\n"
            )
        history_block = (
            f"【⚠️ 故事歷程記錄 — 這是你必須記住的完整故事背景，不可遺忘或自相矛盾】\n"
            f"以下是故事至今每一章的關鍵進展。新章節必須在此基礎上延續，"
            f"所有已發生的事件、已確立的關係、已做出的決策，在後續章節中必須持續有效：\n\n"
            f"{overview_section}"
            f"{lines}\n\n"
            f"【禁止】在不同章節中讓相同情節或對話重複發生；"
            f"【禁止】無故換地點——若需移動場景，必須在正文中明確交代移動過程。"
        )

    env_block = f"【環境設定】\n{environment}" if environment else ""

    bible_block = ""
    if story_bible:
        parts = []
        if story_bible.get("banned_phrases"):
            # Cap to most recent 30 to avoid prompt bloat
            phrases = "\n".join(f"- {p}" for p in story_bible["banned_phrases"][-30:])
            parts.append(f"【絕對禁止重複使用的語句（逐字禁用）】\n{phrases}")
        if story_bible.get("used_tropes"):
            # Cap to most recent 20 to avoid prompt bloat
            tropes = "\n".join(f"- {t}" for t in story_bible["used_tropes"][-20:])
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
            facts = "\n".join(f"- {f}" for f in story_bible["established_facts"] if isinstance(f, str))
            parts.append(
                f"【⚠️ 已確立的故事事實 — 必須全部記住，絕對不可違背】\n"
                f"以下每一條都是前面章節已明確建立的事實，包括角色位置、關係現況、能力狀態、重要物品。\n"
                f"後續章節不得推翻、忽略、遺忘或與之矛盾，違反任何一條即視為失敗：\n{facts}"
            )
        if story_bible.get("asked_questions"):
            recent_qs = story_bible["asked_questions"]  # all questions, no cap
            qs_str = "\n".join(f"- {q}" for q in recent_qs)
            parts.append(
                f"【⚠️ 已問過的問題 — 嚴禁重複】\n"
                f"以下問題在前面章節已經問過，本章任何角色都不可以再問相同或意思相近的問題：\n"
                f"{qs_str}"
            )
        if story_bible.get("paragraph_starters"):
            # Inject most recent 70 starters — prevent verbatim paragraph reuse across chapters
            recent_starters = story_bible["paragraph_starters"][-70:]
            starters_str = "\n".join(f"- {s}…" for s in recent_starters)
            parts.append(
                f"【⚠️ 全故事已使用的段落開頭 — 最高禁令，逐條硬性執行】\n"
                f"以下每一條都是前面章節真實出現過的段落開頭。\n"
                f"本章任何段落的開頭，都不可與以下任何一條相同或高度相似（包括只換一兩個字的版本）。\n"
                f"違反即視為重複，必須重寫：\n{starters_str}"
            )
        if story_bible.get("removed_characters"):
            removed = "\n".join(f"- {n}" for n in story_bible["removed_characters"])
            parts.append(
                f"【⛔ 已移除的角色 — 後續章節絕對禁止出現】\n"
                f"以下角色已被用戶從故事中移除，必須當作這些角色從未存在：\n"
                f"禁止出場、被提及、被其他角色談論，或以任何方式出現在劇情中：\n"
                f"{removed}"
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

    _me_parts = ["【⚠️ 主角設定 — 最高優先級，全程嚴格遵守，任何行為、反應、外貌描寫都必須符合以下設定，違反即失敗】"]
    _me_parts.append(f"姓名：{name}　性別：{gender}")
    if appearance:
        _me_parts.append(f"外貌：{appearance}")
    if personality:
        _me_parts.append(f"個性：{personality}（說話方式、行為模式、情緒反應必須全程與此一致，不可前後矛盾）")
    if residence:
        _me_parts.append(f"住所：{residence}")
    if background:
        _me_parts.append(f"背景故事：{background}（影響主角的動機、反應與決策，必須貫穿全文，不可忽略）")
    me_line = "\n".join(_me_parts)

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

    pacing_block = ""
    if pacing:
        pacing_desc = _PACING_GUIDE.get(pacing, "")
        pacing_block = (
            f"【敘事節奏：{pacing} — 整篇必須貫徹】\n"
            f"{pacing_desc}\n"
            f"- 每進入新場景前，先判斷這個場景應快還是慢，再決定細節密度與情緒呈現方式\n"
            f"- 節奏錯誤（快的地方拖沓、慢的地方草率）視為失敗，必須重寫"
        )

    love_tone_block = ""
    if love_tone and cp_type == "我 × 角色":
        tone_desc = _TONE_GUIDE.get(love_tone, "")
        love_tone_block = (
            f"【戀愛情緒基調：{love_tone} — 整篇必須貫徹】\n"
            f"{tone_desc}\n"
            f"- 所有與感情相關的場景、對話、內心描寫，都必須服務這個基調\n"
            f"- 基調是整體方向，不是每一秒都要強調；在日常互動中自然滲透，而非硬套"
        )

    length_block = (
        f"【⚠️ 字數強制規定 — 必須達標，否則視為失敗】\n"
        f"- 本章目標字數：{target} 字，必須完整寫滿，不可草率收尾或提前結束\n"
        f"- 禁止用幾句話帶過場景；每個時刻都要完整展開，細節豐富，直到達標為止\n"
        f"- 字數不足即視為失敗，必須繼續寫直到達到 {target} 字\n"
        f"- 禁止靠重複同一句型模板（如「她能感覺到X——Y，在Z擴散」）堆積字數；重複句型不計入有效字數，必須用新的角度、新的句子結構補足"
    )

    style_block = ""
    if style_reference:
        style_block = (
            "【⚠️ 文風強制執行 — 最高優先級，每一句都必須貫徹，省略即失敗】\n"
            "根據以下文風分析，用完全相同的密度、節奏與選字精準度寫作：\n\n"
            f"{style_reference}\n\n"
            "【逐條硬性規則，違反任何一條即視為失敗】\n"
            "- 嚴禁模糊或抽象詞彙；每個感受用具體可感知的詞語表達（觸覺、溫度、氣味、聲音、視覺細節）\n"
            "- 每個場景必須完整展開描寫，絕對禁止一句話帶過任何動作或反應\n"
            "- 凡覺得可以略過之處，正是必須大篇幅展開的地方，不可有任何省略\n"
            "- 嚴禁「隨後」「不久後」「過了一會兒」「稍後」「片刻後」「沒多久」等跳躍詞；必須寫出完整過程\n"
            "- 情緒透過身體感受呈現（心跳、呼吸、皮膚反應、肌肉張力），嚴禁直接陳述情緒\n"
            "- 動詞必須有力精準，嚴禁「走」「說」「看」「感到」「覺得」等平淡動詞\n"
            "- 【破折號密度 — 硬性上限】每個自然段最多使用 3 個破折號（——）；"
            "禁止在形容詞與被修飾詞之間、動詞與受詞之間、連續動作之間插入——；"
            "破折號只用於句子中刻意製造停頓的關鍵轉折點，或需要讓讀者在此停留的瞬間。\n"
            "  ✗ 錯誤：「她的手——收緊——抓住我的頭髮——微微——用力——像在——回應我。」\n"
            "  ✓ 正確：「她的手收緊，抓住我的頭髮，微微用力。」"
        )

    setting_block = "\n".join(filter(None, [
        training_block,
        _world_str(world_mode, world_input),
        env_block,
        geo_block,
        notes_block,
        facts_block,
        name_ref_block,
        _cp_str(cp_type, cp_characters, name, nsfw),
        pacing_block,
        love_tone_block,
        me_line,
        _extra_chars_str(extra_characters),
        _must_appear_block(extra_characters, cp_characters, cp_type),
        plot_block,
    ]))

    # memory_block is placed right before context so AI reads it last before writing
    memory_block = "\n\n".join(filter(None, [history_block, bible_block]))

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
        f"敘事視角：第三人稱。旁白全程以主角名字「{name}」敘述，嚴禁在旁白中使用「我」作為主語。"
        f"「我」只能出現在角色的對話引號之內；旁白遇到主角一律寫「{name}」，不可寫「我」。"
        f"這是最高優先規則，違反即視為失敗，必須重寫。"
    )

    outline_block = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"【⚠️ 本章大綱 — 最高強制指令，必須嚴格按此推進，不可偏離】\n"
        f"以下大綱的每一個情節點都必須在正文中具體發生，不可省略、合併或用旁白帶過：\n\n"
        f"{outline}\n\n"
        f"【大綱執行規則】\n"
        f"- 大綱描述的場景、事件、互動必須全部出現\n"
        f"- 情節順序依大綱順序推進，不可跳過或調換\n"
        f"- 感情線進展不可超出大綱描述的程度\n"
        f"- 本章結尾必須與大綱的「本章結尾」一致\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if outline.strip() else ""
    )

    directive_block = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"【⚠️ 本章強制執行指示 — 凌駕所有規則，寫作前逐條確認，每條必須在正文中明確體現】\n"
        f"{chapter_directive}\n"
        f"【上列每一條指示必須在本章正文中具體發生；不可用旁白帶過、暗示、或跳過；任何一條未執行即視為失敗。】\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if chapter_directive.strip() else ""
    )

    _nsfw_reminder = (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "【🔞 限制級寫作最終確認 — 動筆前逐條核對，全部必須執行，違反任何一條即失敗】\n"
        "① 五個階段缺一不可，每階段最少 200 字：前戲→升溫→對話心理→高潮→餘韻\n"
        "② 高潮必須寫全：顫抖部位＋聲音音色＋液體（量/溫度/流動）＋意識狀態變化，只寫「顫抖幾秒然後放鬆」即失敗\n"
        "③ 餘韻最少 150 字：液體殘留、身體狀態、氣息、情緒，不可只有兩句話\n"
        "④ 身體部位用具體情色詞彙，每個部位描寫包含外觀＋質感＋當下狀態三項\n"
        "⑤ 若本章兩個角色輪流（先後口交/先後高潮）：\n"
        "   後一段禁用前一段的任何比喻、句型骨架、動詞組合，必須從完全不同的角度切入\n"
        "⑥ 禁止在同一章重複使用：「像在演奏...樂器/旋律」「從喉嚨深處發出...呻吟」\n"
        "   「不是推開，而是把她拉得更近」「身體猛地繃緊...手指滑落在床單上」等模板\n"
        "⑦ 文風：親密場景的選字、節奏、動詞密度必須與本章其他段落完全一致；\n"
        "   若設定了文風分析，其規則在NSFW段落同樣有效，不可退化為公式化描寫；\n"
        "   讀者不應感覺到從普通段落進入親密段落時風格突然改變\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if nsfw else ""
    )

    if chapter_num == 1:
        _outline_final_reminder_ch1 = (
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"【🔴 動筆前最終確認：本章大綱 — 以下每個情節點必須在正文中具體發生，不可省略】\n"
            f"{outline.strip()}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            if outline.strip() else ""
        )
        prompt = f"""{outline_block + chr(10) + chr(10) if outline_block else ""}{directive_block + chr(10) + chr(10) if directive_block else ""}故事設定：
{setting_block}

輸出語言：{language}
{pov_instruction}
{progress_note}

【我的背景故事】
{background}

請創作第一章，建立世界氛圍與角色，帶出故事開端{"，給出完整結局。" if is_final else "，結尾留下讓人想繼續讀的鉤子。"}
{length_block}
{style_block}
{outline_block}
{directive_block}{_nsfw_reminder}{_outline_final_reminder_ch1}
直接輸出故事正文，格式如下：

第一章　[章節標題]

（正文）"""
    else:
        if len(prev_chapter_text) > 3500:
            context = (
                prev_chapter_text[:900]
                + "\n\n[……前章中段省略，以上為開頭，以下為結尾……]\n\n"
                + prev_chapter_text[-2400:]
            )
        else:
            context = prev_chapter_text
        ending_instruction = (
            "請將故事帶向完整結局，收束所有主線與情感線，給讀者滿足感。"
            if is_final else
            "請自然銜接上一章，推進情節，帶出新的發展或衝突，結尾留下鉤子。"
        )
        memory_header = (
            f"【⚠️ 故事記憶 — 在讀任何其他內容之前，先完整讀完以下所有記錄，寫作時必須全部遵守】\n"
            f"{memory_block}\n"
            f"【以上記憶讀完，方可繼續閱讀下方設定與規則】\n"
        ) if memory_block else ""
        _outline_final_reminder = (
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"【🔴 動筆前最終確認：本章大綱 — 以下每個情節點必須在正文中具體發生，不可省略】\n"
            f"{outline.strip()}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            if outline.strip() else ""
        )
        prompt = f"""{memory_header}
{outline_block + chr(10) if outline_block else ""}{directive_block + chr(10) if directive_block else ""}故事設定：
{setting_block}

輸出語言：{language}
{pov_instruction}
{progress_note}

{ending_instruction}
{length_block}
{style_block}
{outline_block}
{directive_block}
【⚠️ 上一章結尾 — 強制執行以下兩條，違反即失敗】
1. 本章開場必須直接承接以下最後一幕的時間點與地點，不可跳過或無視
2. 角色所在地點與上一章結尾一致；若需換場景，必須在正文中明確交代移動過程
{context}
{_nsfw_reminder}{_outline_final_reminder}
直接輸出故事正文，格式如下：

第{chapter_num}章　[章節標題]

（正文）"""

    training_block_sys = notes_to_prompt_block(training_notes or [])
    style_sys_addon = (
        f"\n\n【文風強制規則 — 凌駕所有其他規則，包含限制級場景，整篇每句話都必須符合】\n"
        f"用戶提供的文風分析如下，必須完全照此密度、節奏、選字精準度寫作；"
        f"此規則對親密場景與非親密場景具有完全相同的效力，不可在NSFW段落退化為模板：\n{style_reference.strip()}\n"
        f"核心原則：動詞承載張力、形容詞僅在關鍵處點睛、感官細節以觸覺與視覺為主、"
        f"時間以連續動作推進（禁止跳躍詞）、情緒透過身體反應而非心理描述。"
        if style_reference.strip() else ""
    )
    directive_sys_prefix = (
        f"【⚠️ 本章強制執行指示 — 最高優先級，凌駕以下所有規則】\n"
        f"{chapter_directive.strip()}\n"
        f"上列指示必須逐條在正文中具體體現，不可省略或用旁白帶過。\n\n"
        if chapter_directive.strip() else ""
    )
    cp_guard = _cp_chapter_guard(cp_characters, chapter_num, cp_type)
    protagonist_guard = ""
    if name and (personality or appearance or background):
        _pg = ["【⚠️ 主角角色設定 — 本章最高優先級，以下每條不可違背，違反任何一條即視為失敗】"]
        if personality:
            _pg.append(f"- 主角個性：{personality}。全程保持此個性，說話語氣、行為模式、情緒反應必須與此完全一致，不可前後矛盾。")
        if appearance:
            _pg.append(f"- 主角外貌：{appearance}。每次描寫主角外貌時嚴格符合此設定，不可自行更改或補充。")
        if background:
            _pg.append(f"- 主角背景：{background}。主角的動機、反應、決策必須符合此背景，不可忽略或違背。")
        protagonist_guard = "\n".join(_pg)
    system_content = (
        (cp_guard + "\n\n" if cp_guard else "")
        + (protagonist_guard + "\n\n" if protagonist_guard else "")
        + directive_sys_prefix
        + _SYSTEM
        + (_NSFW_ADDON if nsfw else "")
        + (_MANIFESTATION_ADDON if world_mode == "顯化日記" else "")
        + _STYLE_SYSTEM_ADDON
        + (f"\n\n{training_block_sys}" if training_block_sys else "")
        + style_sys_addon
    )

    stream = client.chat.completions.create(
        model=_model_for(client),
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=8000,
        temperature=0.82,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
