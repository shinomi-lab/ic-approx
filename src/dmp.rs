use std::ops::Mul;
use std::{mem, time::Instant};

use bitvec::vec::BitVec;
use faer::{Col, Mat};

use crate::InfluenceData;

#[derive(Debug)]
pub struct Params {
    pub steps: usize,
}

pub fn finite_dmp(
    nnodes: usize,
    edges: &[(usize, usize)],
    preds_of: &[Vec<usize>],
    prob: &Mat<f64>,
    seeds: &BitVec,
    params: &Params,
) -> InfluenceData {
    let instant = Instant::now();

    let mut q_curr = Mat::<f64>::zeros(nnodes, nnodes);
    for &(i, j) in edges {
        q_curr[(i, j)] = f64::from(seeds[i]); // true -> 1.0, false -> 0.0
    }
    let mut q_next = Mat::<f64>::zeros(nnodes, nnodes);

    for _ in 0..params.steps {
        for &(j, i) in edges {
            let temp = preds_of[j]
                .iter()
                .filter_map(|&l| {
                    if l == i {
                        None
                    } else {
                        Some(1.0 - prob[(l, j)] * q_curr[(l, j)])
                    }
                })
                .fold(f64::from(!seeds[j]), f64::mul);
            q_next[(j, i)] = 1.0 - temp;
            mem::swap(&mut q_curr, &mut q_next);
        }
    }

    let q = Col::<f64>::from_fn(nnodes, |i| {
        let temp = preds_of[i]
            .iter()
            .map(|&j| 1.0 - prob[(j, i)] * q_curr[(j, i)])
            .fold(f64::from(!seeds[i]), f64::mul);
        1.0 - temp
    });
    InfluenceData::new(q, instant.elapsed())
}

#[cfg(test)]
mod tests {
    use bitvec::{bits, prelude::Lsb0, vec::BitVec};
    use faer::Mat;

    use super::{Params, finite_dmp};
    use crate::graph::preds_of;

    #[test]
    fn test_finite_dmp() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let nnodes = 4;
        let prb = {
            let mut prb = Mat::<f64>::zeros(nnodes, nnodes);
            for &e in &edges {
                prb[e] = 0.5;
            }
            prb
        };
        let seeds = BitVec::from_bitslice(&bits![0, 1, 0, 0]);
        let preds_of = preds_of(nnodes, &edges);

        let data = finite_dmp(
            nnodes,
            &edges,
            &preds_of,
            &prb,
            &seeds,
            &Params { steps: 2 },
        );

        assert_eq!(data.distr[0], 0.0);
        assert_eq!(data.distr[1], 1.0);
        assert_eq!(data.distr[2], 0.5);
        assert_eq!(data.distr[3], 0.25);
    }
}
