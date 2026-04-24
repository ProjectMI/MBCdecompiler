ЗАПУСК ПРОЕКТА

1. Построить HIR для одного скрипта:

python analyze_mbc.py имя_скрипта.mbc

По умолчанию анализируются все function entry points из таблицы definitions плюс export-only записи.
Экспорты, у которых уже есть definition с точным span, не анализируются второй раз — export metadata
сохраняется в `body_selection.entry`, но тело берётся из definition.

Файлы по умолчанию:
- `hir/<script>.hir.json` — JSON с canonical instructions, CFG, dataflow, HIR и отчётами.
- `hir/<script>.hir.txt` — человекочитаемый HIR.

2. Компактный JSON без листинга canonical/HIR details:
python analyze_mbc.py имя_скрипта.mbc --summary-only --out out.json

3. Режим только по definitions:
python analyze_mbc.py имя_скрипта.mbc --definitions-only --summary-only --out definitions.json

4. Прогон по всему корпусу .mbc:
python analyze_mbc.py --out report.json

6. Полная валидация корпуса отдельным флагом:
python analyze_mbc.py --validate-corpus --out report.validation.json