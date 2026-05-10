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
    """Analyze a writing sample and return actionable style rules with concrete examples."""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
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


def fix_repetitive_paragraphs(client: OpenAI, chapter_text: str) -> str:
    """Detect verbatim repeated paragraph blocks and rewrite the duplicates."""
    paragraphs = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]
    seen: dict[str, int] = {}
    has_duplicates = False
    for para in paragraphs:
        key = para[:40]
        if key in seen:
            has_duplicates = True
            break
        seen[key] = 1
    if not has_duplicates:
        return chapter_text
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content":
            "以下章節中存在逐字重複的段落（同一段文字出現兩次以上）。\n"
            "請找出所有重複的段落，將第二次及之後出現的改寫成推進場景的新內容：\n"
            "- 動作要有變化：力道、節奏、角度、深度、速度至少改變一項\n"
            "- 兩人的反應要有層次推進：情緒升溫、身體變化、對話內容演進\n"
            "- 不可只換幾個詞，必須寫出真正不同的場景發展\n"
            "- 第一次出現的段落完全保留不動\n"
            "- 直接輸出修改後的完整章節正文，不加任何說明或標記\n"
            "- 必須輸出完整全文，不可截斷或省略\n\n"
            f"{chapter_text}"
        }],
        max_tokens=8000,
        temperature=0.5,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else chapter_text


def fix_sensory_crutches(client: OpenAI, chapter_text: str) -> str:
    """Rewrite overused 她能X到那股Y sensory perception sentences into direct descriptions."""
    import re
    count = len(re.findall(r'她能[^\s，。！？]{1,4}到', chapter_text))
    if count <= 3:
        return chapter_text
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
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
            "- 直接輸出修改後的完整章節正文，不加任何說明或標記\n"
            "- 必須輸出完整全文，不可截斷\n\n"
            f"{chapter_text}"
        }],
        max_tokens=8000,
        temperature=0.3,
    )
    result = resp.choices[0].message.content.strip()
    return result if result else chapter_text


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
            "2. 直接輸出修正後的完整章節正文，不要加說明或標記\n"
            "3. 必須輸出完整全文，不可截斷、省略或縮短任何段落；若原文很長，也必須完整輸出到最後一個字\n\n"
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

【限制級描寫密度 — 每章必須維持第一章的細節水準，不可退化】
- 描寫身體部位時必須使用具體的情色詞彙（如：肉莖、嫩穴、乳尖、乳暈、菊穴、龜頭、莖身等），嚴禁以「那根東西」「那裡」「它」等模糊代稱替代
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
- 【高潮段落必須各自不同】同一章中若出現多次高潮，每次的描寫方式、句型、視角、側重點必須截然不同；禁止複製貼上相同的高潮框架（如「Dalon Tsai的身體在顫抖，陰道在收縮，在顫抖。然後——那股噴射。」）；第一次可寫生理細節，第二次可寫角色反應與心理，第三次可寫環境或聲音，必須真正推進，而非重播
- 【動作動詞必須輪換】同一個持續動作（如抽插、吸吮、磨蹭）禁止在三段之內使用相同的動詞組合；必須交替使用不同角度的動詞：力道變化、節奏轉換、角度調整、情緒層次，讓讀者感受到動態的推進而非靜止的循環
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
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=8000,
        temperature=0.72,
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
        # Only pass the most recent 5 summaries to prevent prompt bloat
        recent = prev_summaries[-5:]
        start_idx = len(prev_summaries) - len(recent)
        lines = "\n".join(f"第 {start_idx+i+1} 章：{s}" for i, s in enumerate(recent))
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
            "- 動詞必須有力精準，嚴禁「走」「說」「看」「感到」「覺得」等平淡動詞"
        )

    setting_block = "\n".join(filter(None, [
        training_block,
        _world_str(world_mode, world_input),
        env_block,
        geo_block,
        notes_block,
        name_ref_block,
        _cp_str(cp_type, cp_characters, name, nsfw),
        pacing_block,
        love_tone_block,
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
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"【⚠️ 本章強制執行指示 — 凌駕所有規則，寫作前逐條確認，每條必須在正文中明確體現】\n"
        f"{chapter_directive}\n"
        f"【上列每一條指示必須在本章正文中具體發生；不可用旁白帶過、暗示、或跳過；任何一條未執行即視為失敗。】\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if chapter_directive.strip() else ""
    )

    if chapter_num == 1:
        prompt = f"""{directive_block + chr(10) + chr(10) if directive_block else ""}故事設定：
{setting_block}

輸出語言：{language}
{pov_instruction}
{progress_note}

【我的背景故事】
{background}

請創作第一章，建立世界氛圍與角色，帶出故事開端{"，給出完整結局。" if is_final else "，結尾留下讓人想繼續讀的鉤子。"}
{length_block}
{style_block}
{directive_block}
直接輸出故事正文，格式如下：

第一章　[章節標題]

（正文）"""
    else:
        context = prev_chapter_text[-3000:] if len(prev_chapter_text) > 3000 else prev_chapter_text
        ending_instruction = (
            "請將故事帶向完整結局，收束所有主線與情感線，給讀者滿足感。"
            if is_final else
            "請自然銜接上一章，推進情節，帶出新的發展或衝突，結尾留下鉤子。"
        )
        prompt = f"""{directive_block + chr(10) + chr(10) if directive_block else ""}故事設定：
{setting_block}

輸出語言：{language}
{pov_instruction}
{progress_note}

【上一章結尾】
{context}

{ending_instruction}
{length_block}
{style_block}
{directive_block}
直接輸出故事正文，格式如下：

第{chapter_num}章　[章節標題]

（正文）"""

    training_block_sys = notes_to_prompt_block(training_notes or [])
    style_sys_addon = (
        f"\n\n【文風強制規則 — 凌駕所有其他規則，每句話都必須符合】\n"
        f"用戶提供的文風分析如下，必須完全照此密度、節奏、選字精準度寫作：\n{style_reference.strip()}\n"
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
    system_content = (
        directive_sys_prefix
        + _SYSTEM
        + (_NSFW_ADDON if nsfw else "")
        + _STYLE_SYSTEM_ADDON
        + (f"\n\n{training_block_sys}" if training_block_sys else "")
        + style_sys_addon
    )

    stream = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=8000,
        temperature=0.72,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
