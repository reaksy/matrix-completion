# Краткое руководство по синтетическим экспериментам

## Главные файлы

- `synthetic_api.py` — основной файл с функциями.
- `synthetic_control_panel.ipynb` — тетрадь Jupyter для ручного запуска опытов.

## Как устроена работа

Синтетический опыт строится по шагам:

1. создается исходная матрица;
2. при необходимости к ней добавляется дополнительная структура;
3. строится поле пропусков;
4. из него получается маска наблюдаемых элементов;
5. наблюдаемые элементы делятся на обучающую, проверочную и тестовую части;
6. добавляется шум;
7. все собирается в объект `Scenario`;
8. на этом объекте запускаются методы;
9. считаются ошибки, строятся таблицы и рисунки.

## Основные настройки

Все собирается через `ScenarioConfig`.

Внутри него есть:

- `matrix` — размер, ранг, вид матрицы;
- `structure` — дополнительная структура;
- `missingness_field` — вид пропусков;
- `mask_sampling` — доля наблюдаемых элементов;
- `split` — доли проверочной и тестовой частей;
- `noise` — шум.

## Самый простой путь

1. Открыть `synthetic_control_panel.ipynb`
2. Изменить нужные настройки
3. Выполнить ячейки по порядку
4. Сначала посмотреть на матрицу и маски
5. Потом запускать методы

## Полезные функции

### Сборка одного сценария

- `generate_base_matrix(...)`
- `apply_structure(...)`
- `build_missingness_field(...)`
- `sample_observed_mask(...)`
- `split_observed_mask(...)`
- `build_observed_matrix(...)`
- `package_scenario(...)`

### Запуск методов

- `make_method(...)`
- `run_experiment_on_scenario(...)`
- `run_single_experiment(...)`

Примеры методов:

- `make_method('soft_impute', rank=4, max_iter=80)`
- `make_method('als', rank=4, max_iter=50, reg=1e-3)`
- `make_method('rgd', rank=4, max_iter=120)`
- `make_method('rgd_l2', rank=4, max_iter=120, l2_reg=0.03)`
- `make_method('compact_rgd', rank=4, max_iter=100)`

### Перебор одного параметра

- `one_factor_sweep(base_config, parameter_path, values, methods, seeds)`

Примеры путей:

- `'matrix.rank'`
- `'missingness_field.mode'`
- `'matrix.coherence_mode'`
- `'noise.std'`

## Как получить таблицы

После одного запуска:

- `result.records`

После нескольких запусков:

- `records = collect_records(runs)`
- `summary = aggregate_records(records, by=['method'])`

Или, например:

- `summary = aggregate_records(records, by=['matrix.rank', 'method'])`

## Как строить рисунки

Готовые функции:

- `plot_matrix(...)`
- `plot_mask(...)`
- `plot_metric(...)`
- `plot_histories(...)`

Обычно сначала полезно посмотреть:

- истинную матрицу;
- поле пропусков;
- маску наблюдений;
- ход убывания ошибки по итерациям.

## Как сохранять результаты

- `write_csv(...)`
- `write_latex_table(...)`

То есть можно сразу сохранить:

- исходные результаты;
- сводную таблицу;
- таблицу для `LaTeX`.

## Что использовать сейчас

Для новых опытов лучше брать:

- `synthetic_api.py`
- `synthetic_control_panel.ipynb`

## Если хочется начать с простого

Удобный первый опыт:

- матрица `60 x 60`;
- `rank = 4`;
- случайные пропуски;
- `observed_fraction = 0.35`;
- `noise std = 0.02`;
- методы: `soft_impute`, `rgd`, `rgd_l2`.

После этого уже удобно по одному менять:

- ранг;
- вид пропусков;
- уровень шума;
- степень когерентности.
