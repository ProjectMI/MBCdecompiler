#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
if_postfix_stats.py

Скрипт построчно читает указанный файл, находит строки, начинающиеся с "if "
(игнорируя лидирующие пробелы), извлекает следующее слово как постфикс и
собирает статистику по частоте встречаемости каждого постфикса.

Нормализация постфикса в "категорию":
— усекается всё, что идёт после первой скобки из набора ()[]{}<>
— по умолчанию удаляются замыкающие цифры в конце базового имени (индекс)
Опционально можно сохранить цифры флагом --keep-digits.

Упрощённая адресация пути под типовую структуру:
корневой каталог проекта содержит папки "tools" и "mbc", скрипт лежит в "tools",
входные файлы — в "mbc". Если передать только имя файла или относительный путь,
скрипт автоматически попробует найти его в ../mbc относительно местоположения самого скрипта.
Порядок поиска: путь как задан, затем ../mbc/<путь>, затем каталог скрипта/<путь>.
При необходимости базу "mbc" можно переопределить ключом --mbc-root.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple


BRACKETS = '()[]{}<>'
BRACKETS_SET = set(BRACKETS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Построчный разбор файла: учитываются только строки, начинающиеся с "if ", '
            'следующее слово после "if " трактуется как постфикс; строится частотная статистика '
            'с нормализацией категории и удобным поиском файла в ../mbc.'
        )
    )
    p.add_argument('path', help='Путь или имя входного файла. Можно указать просто имя, если файл лежит в ../mbc.')
    p.add_argument(
        '--mbc-root',
        default=None,
        help='Необязательный путь к каталогу mbc, если он отличается от расположенного рядом со скриптом.'
    )
    p.add_argument(
        '--encoding',
        default='utf-8',
        help='Кодировка входного файла (по умолчанию utf-8).'
    )
    p.add_argument(
        '--insensitive',
        action='store_true',
        help='Нормализовать регистр категорий (привести к нижнему регистру).'
    )
    p.add_argument(
        '--keep-digits',
        action='store_true',
        help='Не удалять замыкающие цифры после усечения по скобке.'
    )
    p.add_argument(
        '--csv',
        dest='csv_path',
        help='Опциональный путь для выгрузки результатов в CSV с колонками category,count.'
    )
    return p.parse_args()


def normalize_category(token: str, normalize_lower: bool, keep_digits: bool) -> Optional[str]:
    if not token:
        return None

    # Усекаем по первой скобке
    cut_pos = None
    for i, ch in enumerate(token):
        if ch in BRACKETS_SET:
            cut_pos = i
            break
    if cut_pos is not None:
        token = token[:cut_pos]

    token = token.strip().strip(',:;')

    if not token:
        return None

    if not keep_digits:
        token = re.sub(r'\d+$', '', token)

    token = token.strip('_-')
    if not token:
        return None

    if normalize_lower:
        token = token.lower()

    return token


def iter_postfixes(path: Path, encoding: str, normalize_lower: bool, keep_digits: bool):
    with path.open('r', encoding=encoding, errors='replace') as fh:
        for raw in fh:
            s = raw.lstrip()
            if not s.startswith('if '):
                continue
            parts = s.split(maxsplit=2)
            if len(parts) < 2:
                continue
            token = parts[1]
            cat = normalize_category(token, normalize_lower=normalize_lower, keep_digits=keep_digits)
            if cat:
                yield cat


def resolve_input_path(raw_path: str, mbc_root_opt: Optional[str]) -> Tuple[Optional[Path], List[Path]]:
    """
    Разумная попытка найти файл по нескольким местам, исходя из структуры проекта:
    1) путь как указан пользователем (относительный — относительно текущего каталога запуска)
    2) ../mbc/<path> относительно местоположения самого скрипта
    3) каталог скрипта/<path> на случай локального запуска из tools с относительными ссылками
    4) если передан --mbc-root, то <mbc-root>/<path> тоже проверяется с высоким приоритетом
    """
    raw = Path(raw_path)
    attempts: List[Path] = []

    script_dir = Path(__file__).resolve().parent
    parent = script_dir.parent

    # Пользовательский mbc-root имеет приоритет среди "умных" попыток
    if mbc_root_opt:
        attempts.append(Path(mbc_root_opt) / raw)

    # Как указано пользователем (относительно CWD, если не абсолютный)
    attempts.append(raw if raw.is_absolute() else Path.cwd() / raw)

    # ../mbc/<path> от места расположения скрипта
    attempts.append(parent / 'mbc' / raw)

    # tools/<path> рядом со скриптом
    attempts.append(script_dir / raw)

    for cand in attempts:
        if cand.exists() and cand.is_file():
            return cand, attempts

    return None, attempts


def main() -> int:
    args = parse_args()

    found_path, attempts = resolve_input_path(args.path, args.mbc_root)
    if not found_path:
        sys.stderr.write('Ошибка: файл не найден по ни одному из предполагаемых путей.\n')
        sys.stderr.write('Проверенные варианты:\n')
        for a in attempts:
            try:
                sys.stderr.write(f'  - {a}\n')
            except Exception:
                pass
        return 1

    try:
        counter = Counter(
            iter_postfixes(found_path, args.encoding, args.insensitive, args.keep_digits)
        )
    except UnicodeError as e:
        sys.stderr.write(f'Ошибка декодирования файла: {e}\n')
        return 1

    total_lines = sum(counter.values())
    unique_postfixes = len(counter)

    if total_lines == 0:
        print('Совпадений не найдено. Убедитесь, что строки действительно начинаются с "if " и что кодировка указана верно.')
    else:
        width_key = max((len(k) for k in counter.keys()), default=8)
        header_key = 'category'
        header_val = 'count'
        width_key = max(width_key, len(header_key))

        print(f'Файл: {found_path}')
        print(f'Найдено вхождений: {total_lines}; уникальных категорий: {unique_postfixes}')
        print(f'{header_key:<{width_key}}  {header_val}')
        print('-' * (width_key + 2 + len(header_val)))
        for key, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f'{key:<{width_key}}  {cnt}')

    if args.csv_path:
        out_path = Path(args.csv_path)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['category', 'count'])
                for key, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
                    writer.writerow([key, cnt])
            print(f'\nCSV сохранён: {out_path.resolve()}')
        except OSError as e:
            sys.stderr.write(f'Не удалось записать CSV: {e}\n')
            return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
