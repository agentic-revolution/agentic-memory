use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

fn core_hashmap_insert(c: &mut Criterion) {
    c.bench_function("core_hashmap_insert_1000", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            for i in 0..1000 {
                map.insert(format!("key_{}", i), i);
            }
            black_box(map);
        })
    });
}

fn core_hashmap_lookup(c: &mut Criterion) {
    let mut map = HashMap::new();
    for i in 0..1000 {
        map.insert(format!("key_{}", i), i);
    }
    c.bench_function("core_hashmap_lookup", |b| {
        b.iter(|| {
            black_box(map.get("key_500"));
        })
    });
}

criterion_group!(benches, core_hashmap_insert, core_hashmap_lookup);
criterion_main!(benches);
