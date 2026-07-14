#!/usr/bin/env python3
"""
Профессиональный парсер диалогов для русской прозы (Шолохов).
Обрабатывает:
- «...» (кавычки-ёлочки) внутри авторского текста
- — Реплика, — сказал X. (тире-диалог с inline-атрибуцией)
- Чистые диалоги (— Реплика без атрибуции)
- Разделяет текст реплики и авторскую атрибуцию в одной строке
"""

import json
import re
import os
from collections import Counter

DOC_SLUG = "tihiy-don-ch1-10"


def clean_chapter(text: str) -> str:
    lines = text.split("\n")
    clean = []
    skip = False
    for line in lines:
        line = line.strip()
        if re.search(r'^(Михаил Шолохов|Приглашаем посетить|Тихий Дон|Книга первая|Предыдущая|Оглавление|Следующая|Главная|Раздел сайта|Проза|Поделиться)', line):
            continue
        if re.search(r'^(Гаршин|Островский|Замятин|Гнедич|Салтыков-Щедрин|Дмитриев|Толстой А\.Н\.|Чуковский|Мамин-Сибиряк)\.?$', line):
            continue
        if re.search(r'^\([^)]*\.lit-info\.ru\)$', line):
            continue
        if re.match(r'^\[\d+\]$', line):
            continue
        if re.match(r'^\[', line):
            continue
        if re.match(r'^(Примечания|Предыдущая страница|Оглавление|Следующая страница)', line):
            skip = True
            continue
        if skip:
            if line == "":
                skip = False
            continue
        line = re.sub(r'\s*\[\d+\]', '', line)
        clean.append(line)
    text = "\n".join(clean)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── ПАТТЕРНЫ ────────────────────────────────────────────────────────

# Глаголы речи (для атрибуции)
SPEECH_VERBS = '|'.join([
    'сказал', 'сказала', 'сказали', 'скажу',
    'говорит', 'говорю', 'говорила', 'говорили', 'гутарил', 'гутарили',
    'ответил', 'ответила', 'ответили', 'отвечает',
    'крикнул', 'крикнула', 'крикнули', 'кричит', 'кричал', 'кричала',
    'спросил', 'спросила', 'спросили', 'спрашивает',
    'шепнул', 'шепнула', 'шепчет', 'прошептал', 'прошептала',
    'пробормотал', 'бормотнула', 'бормотал',
    'выдохнул', 'пришептывал',
    'окликнули', 'окликнул', 'обратился', 'обратилась',
    'прикрикнул', 'прикрикнула',
    'рассказывал', 'рассказывает', 'рассказал',
    'осведомился', 'осведомлялся', 'поинтересовался',
    'проговорил', 'проговорила',
    'отозвался', 'отозвалась',
    'ворчала', 'ворчал',
    'усмехнулся', 'усмехнулась', 'улыбнулся', 'улыбнулась',
    'смеялся', 'засмеялась', 'засмеялся',
    'съязвил', 'протянул', 'протянула',
    'покрикивал', 'выкрикивала', 'выкрикнула',
    'закрычал', 'закрычала', 'вскрикнула',
    'гукнул', 'гаркнул',
    'продолжал', 'продолжала',
    'вступилась', 'перебил', 'перебила',
    'повторил', 'повторила',
    'молвил', 'вымолвил',
    'затянул', 'запел', 'запела',
    'позвал', 'позвала',
    'ревел', 'ревела',
    'завел', 'заводит', 'подхватил', 'подхватила',
])

# ── ИЗВЛЕЧЕНИЕ СПИКЕРА ──────────────────────────────────────────────

def extract_speaker(text):
    """Извлекает имя персонажа из текста атрибуции."""
    if not text:
        return None
    
    # Ищем паттерны: ИМЯ + ГЛАГОЛ или ГЛАГОЛ + ИМЯ
    # «сказал Пантелей», «крикнул Григорий», «ответила Аксинья»
    # 
    # "спросил тот" → не имя
    # "крикнул он" → не имя
    
    # Ищем: глагол речи, за которым следует имя собственное
    name_after = re.findall(
        rf'(?:{SPEECH_VERBS})\s+([А-ЯЁ][а-яё]+(?:[ьъ][\w]+)?(?:\s+[А-ЯЁ][а-яё]+(?:[ье]вич|[ье]вна|[ье]ич)?)?)',
        text
    )
    for name in name_after:
        name = name.strip()
        skip = {'тот', 'Он', 'Она', 'Они', 'Это', 'Тот', 'Весь', 'Сам', 'Я', 'Ты', 'Мы',
                'Кто', 'Что', 'Никто', 'Некто', 'Некого'}
        if name not in skip:
            return name
    
    # Ищем: имя, за которым следует глагол речи
    # «Пантелей сказал», «Григорий ответил»
    name_before = re.findall(
        rf'([А-ЯЁ][а-яё]+(?:[ьъ][\w]+)?)\s+(?:{SPEECH_VERBS})',
        text
    )
    for name in name_before:
        name = name.strip()
        skip = {'Он', 'Она', 'Они', 'Тот', 'Весь', 'Сам'}
        if name not in skip:
            return name
    
    return None


# ── НОРМАЛИЗАЦИЯ ИМЁН ──────────────────────────────────────────────

NAME_MAP = {
    'Григорий': 'григорий', 'Гриша': 'григорий', 'Гришка': 'григорий',
    'Гришунька': 'григорий', 'Гришенька': 'григорий',
    'Пантелей': 'пантелей', 'Пантелеймон': 'пантелей',
    'Прокофьевич': 'пантелей', 'Прокофьич': 'пантелей',
    'Прокофич': 'пантелей',
    'Аксинья': 'аксинья', 'Аксюша': 'аксинья', 'Аксютка': 'аксинья',
    'Астахова': 'аксинья',
    'Петро': 'петро', 'Петр': 'петро', 'Петруха': 'петро',
    'Митька': 'митька', 'Митрий': 'митька', 'Митя': 'митька',
    'Коршунов': 'митька', 'Митькин': 'митька',
    'Дуняшка': 'дуняшка', 'Дуняша': 'дуняшка', 'Дунька': 'дуняшка',
    'Дарья': 'дарья', 'Дашка': 'дарья',
    'Степан': 'степан', 'Степа': 'степан', 'Астахов': 'степан',
    'Христоня': 'христоня', 'Христан': 'христоня',
    'Томилин': 'томилин',
    'Федот': 'федот', 'Бодовсков': 'федот',
    'Сергей': 'мохов', 'Платоныч': 'мохов', 'Платонович': 'мохов',
    'Мохов': 'мохов',
    'Алексей': 'алексей', 'Алешка': 'алексей', 'Алешке': 'алексей',
    'Шамиль': 'алексей',
    'Листницкий': 'листницкий',
    'Малашка': 'малашка', 'Фролова': 'малашка',
    'Ильинична': 'ильинична',
    'Мавра': 'мавра',
    'Прокофий': 'прокофий',
    'Люшня': 'люшня',
    'Кузька': 'кузька',
    'Мартин': 'мартин', 'Прохор': 'прохор',
    'Мишка': 'мишка',
    'Томилин': 'томилин',
    'Бабка': 'бабка',
    'Девушка': 'девушка',
}

def norm_name(raw):
    """Нормализует имя из текста в ключ персонажа."""
    if not raw:
        return 'unknown'
    name = raw.strip().rstrip('.!,;:?— ')
    if name in NAME_MAP:
        return NAME_MAP[name]
    # Проверяем окончания 
    for full, short in NAME_MAP.items():
        if name.startswith(full) or full.startswith(name):
            return short
    return name.lower()


# ── ПАРСИНГ ─────────────────────────────────────────────────────────

def parse_line(line):
    """
    Парсит одну строку на сегменты A/D.
    Учитывает сложные inline-конструкции.
    """
    line = line.strip()
    if not line:
        return []
    if not re.search(r'[а-яА-ЯёЁa-zA-Z]', line):
        return []
    
    # Пропускаем строки-заголовки
    if re.match(r'^(I{1,3}|IV|V|VI{0,3}|VII|VIII|IX|X)$', line):
        return []
    if re.match(r'^\*+\s*\*+$', line):
        return []
    
    # ── СЛОЖНЫЙ СЛУЧАЙ: кавычки «...» с атрибуцией внутри строки ──
    # «Текст» → D, остальное → A
    # «Текст, — сказал X, — текст» → D + A + D
    result = []
    
    # Обрабатываем ёлочки
    remaining = line
    while remaining:
        # Ищем открывающую ёлочку
        m = re.search(r'[«„]', remaining)
        if not m:
            break
        start = m.start()
        
        # Текст до ёлочки — авторский
        if start > 0:
            before = remaining[:start].strip()
            if before and re.search(r'[а-яА-ЯёЁ]', before):
                result.append(('A', before))
        
        # От ёлочки до закрывающей
        rest = remaining[start:]
        end_m = re.search(r'[»“…]', rest[1:])
        if end_m:
            speech = rest[1:1 + end_m.start() + 1]
            # Проверяем, содержит ли речь inline-атрибуцию
            # «...» — чистая реплика
            # «..., — сказал X, ...» — реплика с атрибуцией
            
            # Ищем: «реплика, — атрибуция, — реплика»
            inner_parts = re.split(r'——|—\s*([А-Яа-я].*?)—', speech)
            # Упрощённо: если внутри есть —, то это inline-атрибуция
            if '—' in speech:
                # Пытаемся разбить
                # «Реплика, — сказал X. — Продолжение.»
                # или «Реплика, — сказал X.»
                inline_match = re.match(
                    r'(.*?[.!?]?)\s*[,—]{1,2}\s*([А-Яа-яё].*?)(?:\s*[,—]{1,2}\s*(.*))?$',
                    speech, re.DOTALL
                )
                if inline_match:
                    speech_part = inline_match.group(1).strip()
                    attr_part = inline_match.group(2).strip()
                    cont_part = inline_match.group(3)
                    
                    # Определяем, где кончается атрибуция и начинается продолжение
                    # Если после атрибуции ещё есть текст — это продолжение реплики
                    if speech_part and re.search(r'[а-яА-ЯёЁ]', speech_part):
                        result.append(('D', speech_part))
                    
                    if attr_part and re.search(r'[а-яА-ЯёЁ]', attr_part):
                        result.append(('A', attr_part))
                    
                    if cont_part and cont_part.strip():
                        cont = cont_part.strip().rstrip('»') 
                        if cont and re.search(r'[а-яА-ЯёЁ]', cont):
                            result.append(('D', cont))
                else:
                    # Не смогли распарсить — вся реплика целиком D
                    if speech and re.search(r'[а-яА-ЯёЁ]', speech):
                        result.append(('D', speech))
            else:
                # Чистая реплика
                if speech and re.search(r'[а-яА-ЯёЁ]', speech):
                    result.append(('D', speech))
            
            remaining = rest[1 + end_m.start() + 2:]  # после »
        else:
            # Нет закрывающей кавычки — остаток строки
            remaining = ''
            break
    
    # Остаток
    if remaining and remaining.strip() and re.search(r'[а-яА-ЯёЁ]', remaining):
        remaining = remaining.strip()
        if remaining.startswith('»') or remaining.startswith('…»'):
            remaining = remaining.lstrip('»… ')
        if remaining:
            result.append(('A', remaining))
    
    if not result:
        # Не было ёлочек — вся строка либо A, либо D-диалог
        return parse_dash_dialogue(line)
    
    return result


def parse_dash_dialogue(line: str) -> list[dict]:
    """
    Парсит диалог с тире.
    
    Паттерны:
    — Реплика. → D
    — Реплика, — сказал X. → D + A (реплика и атрибуция раздельно)
    — Реплика, — сказал X, — продолжение. → D + A + D
    """
    line = line.strip()
    if not line.startswith('—') and not line.startswith('–'):
        return [('A', line)]
    
    # Убираем начальное тире
    content = re.sub(r'^[—–]\s*', '', line)
    
    # Проверяем на inline-атрибуцию
    # Ищем: реплика, — атрибуция, — продолжение
    # или: реплика, — атрибуция. (конец)
    
    # Разделяем по паттерну: « — атрибуция — » или « — атрибуция.»
    # Если есть второе тире с пробелами, это может быть атрибуция
    parts = re.split(r'\s*[—–]\s*', content)
    
    if len(parts) == 1:
        # Нет inline-атрибуции — чистая реплика
        return [('D', parts[0].strip())]
    
    elif len(parts) == 2:
        # parts[0] — реплика, parts[1] — атрибуция (или наоборот?)
        p1, p2 = parts[0].strip(), parts[1].strip()
        
        # Определяем: если p2 содержит глагол речи — это атрибуция
        if re.search(rf'(?:{SPEECH_VERBS})', p2, re.IGNORECASE):
            # D + A
            result = []
            if p1 and re.search(r'[а-яА-ЯёЁ]', p1):
                result.append(('D', p1))
            if p2 and re.search(r'[а-яА-ЯёЁ]', p2):
                result.append(('A', p2))
            return result
        
        # Если p1 короткое (типа «А?») и p2 длинное — может быть иначе
        # Просто D с тире (часть реплики персонажа)
        return [('D', content.strip())]
    
    else:
        # 3+ части: реплика — атрибуция — продолжение
        p1, p2, *rest = [p.strip() for p in parts]
        result = []
        
        if p1 and re.search(r'[а-яА-ЯёЁ]', p1):
            result.append(('D', p1))
        
        # Атрибуция — содержит глагол речи
        if p2 and re.search(r'[а-яА-ЯёЁ]', p2):
            result.append(('A', p2))
        
        if rest:
            cont = " — ".join(rest).strip()
            # remove trailing period if any
            if cont and re.search(r'[а-яА-ЯёЁ]', cont):
                result.append(('D', cont))
        
        return result


# ── ОСНОВНАЯ СЕГМЕНТАЦИЯ ───────────────────────────────────────────

def segment_text(text: str) -> list[dict]:
    """Сегментирует весь текст на A и D."""
    segments = []
    lines = text.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not re.search(r'[а-яА-ЯёЁa-zA-Z]', line):
            continue
        if re.match(r'^(I{1,3}|IV|V|VI{0,3}|VII|VIII|IX|X)$', line):
            continue
        if re.match(r'^\*+\s*\*+$', line):
            continue
        
        parsed = parse_line(line)
        segments.extend(parsed)
    
    return segments


# ── АТРИБУЦИЯ ───────────────────────────────────────────────────────

def attribute_speakers(segments: list[tuple[str, str]]) -> list[dict]:
    """
    Проставляет speaker для D-сегментов.
    segments: список (type, text)
    returns: список {type, text, speaker}
    """
    # Превращаем в dicts
    result = []
    for t, text in segments:
        result.append({'type': t, 'text': text, 'speaker': None})
    
    # Шаг 1: проходим по всем D, ищем спикера в соседних A-сегментах
    for i, seg in enumerate(result):
        if seg['type'] != 'D':
            continue
        
        speaker = None
        
        # Ищем в ближайшем A слева
        for j in range(i - 1, max(-1, i - 3), -1):
            if result[j]['type'] == 'A':
                s = extract_speaker(result[j]['text'])
                if s:
                    # Проверяем, не совпадает ли с другим методом
                    speaker = s
                break
        
        # Если не нашли — ищем в A справа
        if not speaker:
            for j in range(i + 1, min(len(result), i + 3)):
                if result[j]['type'] == 'A':
                    s = extract_speaker(result[j]['text'])
                    if s:
                        speaker = s
                    break
        
        # Если не нашли — ищем глагол речи в особых конструкциях
        if not speaker:
            # Может быть, это реплика в цепочке диалога
            pass
        
        if speaker:
            seg['speaker'] = norm_name(speaker)
    
    # Шаг 2: диалоговая цепочка (чередование спикеров)
    d_indices = [i for i, s in enumerate(result) if s['type'] == 'D']
    
    for idx in range(1, len(d_indices)):
        prev = d_indices[idx - 1]
        curr = d_indices[idx]
        
        prev_sp = result[prev].get('speaker')
        curr_sp = result[curr].get('speaker')
        
        # Если у текущей реплики нет спикера, но есть у предыдущей
        if not curr_sp and prev_sp:
            # Проверяем длину авторского текста между ними
            between = [result[j] for j in range(prev + 1, curr)]
            between_a = [s['text'] for s in between if s['type'] == 'A']
            between_text = " ".join(between_a)
            
            if not between_text or len(between_text) < 300:
                # Близкие реплики — вероятно, диалог
                # Если между ними нет нового имени — чередуем
                new_name = extract_speaker(between_text) if between_text else None
                if not new_name:
                    # Чередуем спикеров
                    # Ищем все спикеры в ближайшей истории
                    prev_speakers = []
                    for j in range(curr - 1, max(-1, curr - 10), -1):
                        if result[j]['type'] == 'D' and result[j].get('speaker'):
                            prev_speakers.append(result[j]['speaker'])
                    
                    if prev_speakers:
                        # Если последний спикер был = prev_sp, а предпоследний другой
                        # то чередуем. Но если prev_sp повторяется дважды подряд — это тот же
                        if len(prev_speakers) >= 2 and prev_speakers[0] == prev_speakers[1]:
                            # Две реплики одного персонажа подряд — продолжает тот же
                            result[curr]['speaker'] = prev_speakers[0]
                        elif len(prev_speakers) >= 1:
                            # Чередуем с предыдущим
                            if prev_speakers[0] == prev_sp:
                                # Найти другого спикера
                                other_speakers = set(prev_speakers)
                                if len(other_speakers) >= 2:
                                    others = [s for s in other_speakers if s != prev_sp]
                                    if others:
                                        result[curr]['speaker'] = others[0]
        
        # Если у текущей и предыдущей реплик один спикер, но текст между ними
        # указывает на другого — исправляем
        if result[curr].get('speaker') and prev_sp and result[curr]['speaker'] == prev_sp:
            between = [result[j] for j in range(prev + 1, curr)]
            between_a_text = " ".join(s['text'] for s in between if s['type'] == 'A')
            new_name = extract_speaker(between_a_text)
            if new_name:
                nn = norm_name(new_name)
                if nn != prev_sp:
                    result[curr]['speaker'] = nn
    
    # Шаг 3: для оставшихся unknown — контекстная догадка
    # Если реплика в цепочке диалога, а предыдущий говорящий известен
    # и не unknown — используем его же (одна и та же реплика может длиться несколько сегментов)
    
    return result


# ── ВАЛИДАЦИЯ ───────────────────────────────────────────────────────

def merge_adjacent_a(segments: list[dict]) -> list[dict]:
    """Сливает соседние A-сегменты."""
    merged = []
    for seg in segments:
        text = seg['text'].strip()
        if not text or not re.search(r'[а-яА-ЯёЁa-zA-Z0-9]', text):
            continue
        
        if seg['type'] == 'A' and merged and merged[-1]['type'] == 'A':
            merged[-1]['text'] += ' ' + text
        else:
            merged.append(dict(seg))
    
    return merged


def print_stats(segments: list[dict]):
    """Печатает статистику."""
    a = sum(1 for s in segments if s['type'] == 'A')
    d = sum(1 for s in segments if s['type'] == 'D')
    
    speaker_counts = Counter(s.get('speaker', 'unknown') for s in segments if s['type'] == 'D')
    known = {k: v for k, v in speaker_counts.items() if k != 'unknown'}
    unknown = speaker_counts.get('unknown', 0)
    
    print(f"  Сегментов: {len(segments)} ({a}A / {d}D)")
    print(f"  Реплик (D): {d}")
    print(f"  Атрибутировано: {d - unknown}, unknown: {unknown}")
    print(f"  Персонажи ({len(known)}):")
    for sp, cnt in known.most_common(20):
        print(f"    {sp}: {cnt}")


def check_criteria(segments):
    """Проверяет критерии качества."""
    errors = []
    
    d_segs = [s for s in segments if s['type'] == 'D']
    speaker_counts = Counter(s.get('speaker', 'unknown') for s in d_segs)
    
    # ≥2 персонажа с ≥2 репликами
    valid = {sp: cnt for sp, cnt in speaker_counts.items() if cnt >= 2 and sp != 'unknown' and sp}
    if len(valid) < 2:
        errors.append(f"Недостаточно персонажей с ≥2 репликами: {len(valid)} (нужно ≥2)")
    
    # <10% unknown
    total_d = len(d_segs)
    unknown = speaker_counts.get('unknown', 0)
    if unknown / max(total_d, 1) > 0.1:
        errors.append(f"Слишком много unknown: {unknown}/{total_d} = {unknown/total_d*100:.0f}% (макс 10%)")
    
    # Сегменты с пустым speaker для D
    for i, s in enumerate(segments):
        if s['type'] == 'D' and not s.get('speaker'):
            errors.append(f"Сегмент {i} (D) без speaker: {s['text'][:50]}")
            break
    
    return len(errors) == 0, errors


# ── ЗАПУСК ──────────────────────────────────────────────────────────

CHAPTER_FILES = [
    ("ch01.txt", "Глава I"),
    ("ch02.txt", "Глава II"),
    ("ch03.txt", "Глава III"),
    ("ch04.txt", "Глава IV"),
    ("ch05.txt", "Глава V"),
    ("ch06.txt", "Глава VI"),
    ("ch07.txt", "Глава VII"),
    ("ch08.txt", "Глава VIII"),
    ("ch09.txt", "Глава IX"),
    ("ch10.txt", "Глава X"),
]


def process_all():
    raw_dir = "raw_texts"
    all_raw_segments = []  # (type, text)
    
    for fname, title in CHAPTER_FILES:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"⚠️ {fname} не найден")
            continue
        
        text = open(fpath, encoding='utf-8').read()
        text = clean_chapter(text)
        
        # Добавляем заголовок главы
        all_raw_segments.append(('A', f'[{title}]'))
        
        segs = segment_text(text)
        all_raw_segments.extend(segs)
        
        a = sum(1 for t, _ in segs if t == 'A')
        d = sum(1 for t, _ in segs if t == 'D')
        print(f"  {title}: {a}A + {d}D = {len(segs)}")
    
    # Атрибуция
    attributed = attribute_speakers(all_raw_segments)
    
    # Слияние соседних A
    attributed = merge_adjacent_a(attributed)
    
    print(f"\n=== ИТОГО ===")
    print_stats(attributed)
    
    ok, errors = check_criteria(attributed)
    if ok:
        print("✅ Критерии выполнены!")
    else:
        print("⚠️ Критерии НЕ выполнены:")
        for e in errors:
            print(f"  - {e}")
    
    # Запись
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    out_path = os.path.join(data_dir, "speakers_tihiy_don.jsonl")
    with open(out_path, "w", encoding='utf-8') as f:
        for seg in attributed:
            obj = {"doc": DOC_SLUG, "label": seg['type'], "text": seg['text']}
            if seg['type'] == 'D':
                obj['speaker'] = seg.get('speaker', 'unknown')
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"\n📝 Записано {len(attributed)} сегментов в {out_path}")
    
    # Проверка для первого взгляда
    print("\n=== СЕГМЕНТЫ С НЕИЗВЕСТНЫМ СПИКЕРОМ ===")
    unknown = [s for s in attributed if s['type'] == 'D' and s.get('speaker') == 'unknown']
    for s in unknown[:10]:
        print(f'  D(unknown): {s["text"][:80]}...')
    if len(unknown) > 10:
        print(f'  ... и ещё {len(unknown) - 10}')
    
    return attributed


if __name__ == "__main__":
    segs = process_all()
