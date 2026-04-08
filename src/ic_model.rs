use std::time::Duration;
use std::{mem, time::Instant};

use bitvec::{bitvec, vec::BitVec};
use faer::{Col, Mat};
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::InfluenceData;

fn execute_ic_once(
    nnodes: usize,
    adj: &[BitVec],
    prob: &Mat<f64>,
    seeds: &BitVec,
    rng: &mut impl Rng,
) -> (BitVec, usize) {
    let mut t = 0;
    let mut ca = seeds.clone();
    let mut aa = bitvec![0; nnodes];
    let mut next_ca = bitvec![0; nnodes];

    while ca.count_ones() > 0 {
        t += 1;
        next_ca.fill(false);
        for i in 0..nnodes {
            if !ca[i] || aa[i] {
                continue;
            }
            for j in 0..nnodes {
                if !adj[i][j] || ca[j] || aa[j] {
                    continue;
                }
                let p = rng.random::<f64>();
                if prob[(i, j)] > p {
                    next_ca.set(j, true);
                }
            }
        }
        mem::swap(&mut ca, &mut next_ca);
        aa |= &next_ca;
    }
    (aa, t)
}

#[derive(Debug)]
pub struct ExtraData {
    pub mean_steps: f64,
}

pub fn monte_carlo_ic_par(
    nnodes: usize,
    adj: &[BitVec],
    prob: &Mat<f64>,
    seeds: &BitVec,
    rng: &mut impl Rng,
    iter_size: usize,
) -> (InfluenceData, ExtraData) {
    let ncores = num_cpus::get();
    let mut master_rng = Xoshiro256PlusPlus::from_rng(rng);
    let mut rngs = Vec::with_capacity(ncores);

    let k = iter_size / ncores;
    let mut m = iter_size % ncores;

    for i in 0..ncores {
        let r = if m > 0 {
            m -= 1;
            1
        } else {
            0
        };
        rngs.push((master_rng.clone(), k + r));
        if i <= ncores - 1 {
            master_rng.jump();
        }
    }

    let results = rngs
        .par_iter_mut()
        .map(|(rng, k)| {
            (0..*k)
                .map(|_| {
                    let instant = Instant::now();
                    let (aa, t) = execute_ic_once(nnodes, adj, prob, seeds, rng);
                    (aa, t, instant.elapsed())
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    let n = seeds.len();
    let mut v = Col::<f64>::zeros(n);
    let mut sum_t = 0;
    let mut duration = Duration::ZERO;
    for (aa, t, dur) in results {
        for i in aa.iter_ones() {
            v[i] += 1f64;
        }
        sum_t += t;
        duration += dur;
    }
    let m = iter_size as f64;
    v /= m;
    let mean_steps = (sum_t as f64) / m;

    (InfluenceData::new(v, duration), ExtraData { mean_steps })
}

#[cfg(test)]
mod tests {
    use bitvec::{bits, prelude::Lsb0, vec::BitVec};
    use faer::Mat;
    use rand::{SeedableRng, rngs::SmallRng};
    use rand_xoshiro::Xoroshiro128PlusPlus;
    use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

    use super::execute_ic_once;
    use crate::graph::adj_binmat;

    #[test]
    fn test_ic_model() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let nnodes = 4;
        let adj = adj_binmat(nnodes, &edges);
        let prb = {
            let mut prb = Mat::<f64>::zeros(nnodes, nnodes);
            for &e in &edges {
                prb[e] = 1.0;
            }
            prb
        };
        let seeds = BitVec::from_bitslice(&bits![0, 1, 0, 0]);

        let mut rng = SmallRng::seed_from_u64(0);
        let (res, _) = execute_ic_once(nnodes, &adj, &prb, &seeds, &mut rng);
        assert!(!res[0]);
        assert!(res[1]);
        assert!(res[2]);
        assert!(res[3]);
    }

    #[test]
    fn test_stochastic_ic_model() {
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3)];
        let nnodes = 4;
        let adj = adj_binmat(nnodes, &edges);
        let prb = {
            let mut prb = Mat::<f64>::zeros(nnodes, nnodes);
            for &e in &edges {
                prb[e] = 0.5;
            }
            prb
        };
        let seeds = BitVec::from_bitslice(&bits![1, 0, 0, 0]);

        let iter = 100;
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
        let mut rngs = (0..iter)
            .map(|_| {
                rng.jump();
                rng.clone()
            })
            .collect::<Vec<_>>();

        let rss = rngs
            .par_iter_mut()
            .map(|rng| {
                let (res, _) = execute_ic_once(nnodes, &adj, &prb, &seeds, rng);
                res
            })
            .collect::<Vec<_>>();

        let mut total = vec![0usize; nnodes];
        rss.into_iter().for_each(|res| {
            for i in res.iter_ones() {
                total[i] += 1usize;
            }
        });
        println!("{:?}", total);
    }
}
