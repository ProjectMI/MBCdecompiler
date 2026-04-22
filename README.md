ЗАПУСК ПРОЕКТА

1. Построить структурированный HIR для одного скрипта и сохранить JSON + текст:
python analyze_mbc.py имя_скрипта.mbc

По умолчанию будут записаны файлы:
- `hir/<script>.hir.json` — машинно-читаемый payload с canonical instructions, CFG, dataflow и structured HIR.
- `hir/<script>.hir.txt` — человекочитаемый HIR.

2. Построить только компактный JSON без текстового листинга и без canonical details:
python analyze_mbc.py имя_скрипта.mbc --summary-only --out out.json

3. Прогон по всему корпусу `.mbc` без листингов функций. Вместо старого IR-отчёта строится сводка по HIR:
python analyze_mbc.py --out report.json
