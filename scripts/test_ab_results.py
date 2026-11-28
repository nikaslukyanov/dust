import pytest

from ab_results import compare_seeds, summarize_metrics


def test_step_statistics_discard_three_warmup_updates():
    records = [
        dict(step=step, step_host_ms=ms, train_seconds=0.3, draws=50, population=10,
             flops=1000, flops_executed=1400)
        for step, ms in enumerate((100, 90, 80, 6, 2), start=1)
    ]
    result = dict(best_step=4, best_val=3.5, test_at_best_val=3.8, timing={'totals': {'allocation_seconds': 0.2}})
    metrics = summarize_metrics(records, result, 3)
    assert metrics['host_median_ms'] == 4
    assert metrics['host_p10_ms'] == pytest.approx(2.4)
    assert metrics['host_p90_ms'] == pytest.approx(5.6)
    assert metrics['training_seconds'] == 0.3
    assert metrics['allocation_seconds'] == 0.2
    assert metrics['draws'] == 50
    assert metrics['analytic_needed_flops'] == 1000
    assert metrics['analytic_executed_flops'] == 1400


def test_comparisons_are_paired_before_seed_aggregation():
    def reports(base, adaptive):
        return {'fixed': {'metrics': {'host_median_ms': base}},
                'adaptive': {'metrics': {'host_median_ms': adaptive}}}

    seeds = [
        {'seed': 42, 'cases': reports(10, 8)},
        {'seed': 43, 'cases': reports(20, 10)},
        {'seed': 44, 'cases': reports(40, 32)},
    ]
    [comparison] = compare_seeds(seeds, [{'name': 'adaptive', 'control': 'fixed'}])
    assert comparison['case'] == 'adaptive'
    assert comparison['control'] == 'fixed'
    assert [row['delta']['host_median_ms'] for row in comparison['seeds']] == [-2, -10, -8]
    delta = comparison['delta']['host_median_ms']
    assert delta['n'] == 3
    assert delta['mean'] == pytest.approx(-20 / 3)
    assert delta['sample_std'] == pytest.approx((52 / 3) ** 0.5)
    reduction = comparison['reduction_percent']['host_median_ms']
    assert reduction['mean'] == pytest.approx(30)
    assert reduction['sample_std'] == pytest.approx(300 ** 0.5)
