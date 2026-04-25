ЗАПУСК ПРОЕКТА

1. Построить стабильный pre-AST/HIR payload для одного скрипта и AST text:

python analyze_mbc.py имя_скрипта.mbc

По умолчанию анализируются все function entry points из таблицы definitions плюс export-only записи.
Экспорты, у которых уже есть definition с точным span, не анализируются второй раз — export metadata
сохраняется в `body_selection.entry`, но тело берётся из definition.

Файлы по умолчанию:
- `hir/<script>.hir.json` — JSON стабильных фаз до AST: canonical instructions, CFG, dataflow и normalized HIR. Corpus/report-слой здесь намеренно не публикуется.
- `hir/<script>.ast.txt` — человекочитаемый AST/pseudo-source, построенный поверх normalized HIR.

2. Компактный JSON без листинга canonical/HIR details:

python analyze_mbc.py имя_скрипта.mbc --summary-only --out out.json

3. Режим только по definitions:

python analyze_mbc.py имя_скрипта.mbc --definitions-only --summary-only --out definitions.json

4. Прогон по всему корпусу .mbc теперь строит AST report, а не pre-AST/HIR report:

python analyze_mbc.py --out ast-report.json

Формат corpus-отчёта: `ast-report-v1`. Основные секции:
- `summary` — агрегаты по модулям/функциям: structured/fallback block ratios, residual labels/gotos, explicit CFG regions, AST depth/regions/statements/expressions, parameterized blocks, block params, loopbacks.
- `ast_metrics` — гистограммы region/statement/expression kinds и control-flow counters.
- `rankings` — watchlist для следующего rule-based улучшения: худшие fallback-функции, максимальные residual gotos, самые глубокие/большие AST, функции с большим числом structured loopbacks.
- `modules` — компактные summary по каждому MBC-модулю.

5. Полная валидация корпуса отдельным флагом:

python analyze_mbc.py --validate-corpus --out ast-report.validation.json
