use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn stress_10k_strings(c: &mut Criterion) {
    c.bench_function("stress_10k_string_alloc", |b| {
        b.iter(|| {
            let v: Vec<String> = (0..10_000).map(|i| format!("item_{}", i)).collect();
            black_box(v);
        })
    });
}

fn stress_sort_10k(c: &mut Criterion) {
    c.bench_function("stress_sort_10k", |b| {
        b.iter(|| {
            let mut v: Vec<i64> = (0..10_000).rev().collect();
            v.sort();
            black_box(v);
        })
    });
}

criterion_group!(benches, stress_10k_strings, stress_sort_10k);
criterion_main!(benches);
