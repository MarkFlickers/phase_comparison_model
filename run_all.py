"""Воспроизведение всех результатов одной командой:

    python run_all.py            — проверки, опорные числа, серии С1–С6, дополнения к С6
                                   (c6_dbar_common, causal_check), рисунки, сводка
    python run_all.py --quick    — то же с уменьшенным числом испытаний (для проверки установки);
                                   пишет в results_quick/ и figures_quick/

Требуется Python 3.10+ и пакеты numpy, scipy, matplotlib.
Результаты: results/*.csv, figures/*.png и *.pdf, results/summary.md.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as C  # noqa: E402


def main():
    quick = '--quick' in sys.argv
    if quick:
        C.C3_TRIALS = C.C4_TRIALS = C.C6_TRIALS = 40
        C.C2_REPEATS = 1
        C.OUT_SUFFIX = '_quick'  # результаты — в results_quick/ и figures_quick/, полные не трогаем
    t0 = time.time()
    import checks
    import check_reference
    import check_literature
    import series
    import c3_zc_tochnee
    import c6_dbar_common
    import causal_check
    import figures
    import summary
    print('== Проверки корректности ==', flush=True)
    ok = checks.main()
    print('\n== Опорные числа (однофазный расчёт) ==', flush=True)
    check_reference.main()
    print('\n== Сверка с опубликованными данными (фильтр Фурье) ==', flush=True)
    check_literature.main()
    print('\n== Серии С2–С6, С1 ==', flush=True)
    series.main()
    print('\n== Уточнение С3 для перехода через ноль (10 000 испытаний) ==', flush=True)
    if quick:
        c3_zc_tochnee.N_TRIALS = 200
    c3_zc_tochnee.main()
    print('\n== Дополнение С6: неучтённая общая задержка на общем сигнале ==', flush=True)
    c6_dbar_common.main(C.C6_TRIALS)
    print('\n== Причинность МНК: тесты и повтор С6 ==', flush=True)
    causal_check.main(20 if quick else 400)
    print('\n== Рисунки ==', flush=True)
    figures.main()
    print('\n== Сводка ==', flush=True)
    summary.main()
    print(f'\nГотово за {time.time() - t0:.0f} с. Проверки: {"пройдены" if ok else "ЕСТЬ ОШИБКИ"}.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
