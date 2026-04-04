use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use highuvlith_core::aerial::AerialImageEngine;
use highuvlith_core::mask::Mask;
use highuvlith_core::optics::ProjectionOptics;
use highuvlith_core::source::VuvSource;
use highuvlith_core::types::GridConfig;

fn bench_engine_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_creation");
    for size in [64, 128, 256] {
        let source = VuvSource::f2_laser(0.7).unwrap();
        let optics = ProjectionOptics::new(0.75).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let grid = GridConfig {
                size,
                pixel_nm: 2.0,
            };
            b.iter(|| AerialImageEngine::new(&source, &optics, grid.clone(), 20).unwrap());
        });
    }
    group.finish();
}

fn bench_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("aerial_compute");
    for size in [64, 128, 256] {
        let source = VuvSource::f2_laser(0.7).unwrap();
        let optics = ProjectionOptics::new(0.75).unwrap();
        let grid = GridConfig {
            size,
            pixel_nm: 2.0,
        };
        let engine = AerialImageEngine::new(&source, &optics, grid, 20).unwrap();
        let mask = Mask::line_space(65.0, 180.0).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| engine.compute(&mask, 0.0));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_engine_creation, bench_compute);
criterion_main!(benches);
