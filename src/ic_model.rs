use std::time::Duration;
use std::{mem, time::Instant};

use bitvec::{bitvec, vec::BitVec};
use faer::{Col, Mat};
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

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
                if rng.random::<f64>() < prob[(i, j)] {
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
    prb: &Mat<f64>,
    seeds: &BitVec,
    rng: &mut impl Rng,
    iter_size: usize,
) -> (InfluenceData, ExtraData) {
    let mut master_rng = Xoshiro256PlusPlus::from_rng(rng);
    let mut rngs = Vec::with_capacity(iter_size);

    rngs.push(master_rng.clone());
    for _ in 1..iter_size {
        master_rng.jump();
        rngs.push(master_rng.clone());
    }

    let results = rngs
        .par_iter_mut()
        .map(|rng| {
            let instant = Instant::now();
            let (aa, t) = execute_ic_once(nnodes, adj, prb, seeds, rng);
            (aa, t, instant.elapsed())
        })
        .collect::<Vec<_>>();

    let n = seeds.len();
    let mut v = Col::<f64>::zeros(n);
    let mut mean_t = 0f64;
    let mut duration = Duration::ZERO;
    for (aa, t, dur) in results {
        for i in aa.iter_ones() {
            v[i] += 1f64;
        }
        mean_t += t as f64;
        duration += dur;
    }
    let m = iter_size as f64;
    v /= m;
    mean_t /= m;

    (
        InfluenceData::new(v, duration),
        ExtraData { mean_steps: mean_t },
    )
}

#[cfg(test)]
mod tests {
    use bitvec::{bits, prelude::Lsb0, vec::BitVec};
    use faer::Mat;
    use rand::{SeedableRng, rngs::SmallRng};

    use super::execute_ic_once;
    use crate::graph::adj_binmat;

    #[test]
    fn test_simulate_ic_model() {
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
}
