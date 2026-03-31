use std::ops::Mul;
use std::{mem, time::Instant};

use bitvec::vec::BitVec;
use faer::{Col, Mat};

use crate::InfluenceData;

#[derive(Debug)]
pub struct NoselfParams {
    pub steps: usize,
}

pub fn finite_sss_noself(
    nnodes: usize,
    preds_of: &[Vec<usize>],
    prob: &Mat<f64>,
    seeds: &BitVec,
    params: &NoselfParams,
) -> InfluenceData {
    let instant = Instant::now();
    // step s
    let mut curr_probs_temp = Mat::<f64>::zeros(nnodes, nnodes);
    let mut curr_probs = Col::<f64>::zeros(nnodes);

    for j in 0..nnodes {
        let p = f64::from(seeds[j]);
        curr_probs[j] = p;
        for q in seeds.iter_zeros() {
            curr_probs_temp[(q, j)] = p;
        }
    }

    // step s+1
    let mut next_probs_temp = Mat::<f64>::zeros(nnodes, nnodes);
    let mut next_probs = Col::<f64>::zeros(nnodes);

    for _ in 0..params.steps {
        for q in seeds.iter_zeros() {
            for j in 0..nnodes {
                let is_seed = seeds[j];
                next_probs_temp[(q, j)] = if is_seed {
                    1.0
                } else if q == j {
                    0.0 // 1 - prod_i (1 - 0 * pi^q_i) = 1
                } else {
                    1.0 - preds_of[j]
                        .iter()
                        .filter_map(|&i| {
                            if q == i {
                                None
                            } else {
                                Some(1.0 - prob[(i, j)] * curr_probs_temp[(q, i)])
                            }
                        })
                        .reduce(f64::mul)
                        .unwrap_or(1.0)
                };
                next_probs[j] = if is_seed {
                    1.0
                } else {
                    1.0 - preds_of[j]
                        .iter()
                        .map(|&i| 1.0 - prob[(i, j)] * curr_probs_temp[(j, i)])
                        .reduce(f64::mul)
                        .unwrap_or(1.0)
                };
            }
        }
        mem::swap(&mut curr_probs_temp, &mut next_probs_temp);
        mem::swap(&mut curr_probs, &mut next_probs);
    }

    InfluenceData::new(curr_probs, instant.elapsed())
}

#[cfg(test)]
mod tests {
    use bitvec::{bits, prelude::Lsb0, vec::BitVec};
    use faer::Mat;

    use super::{NoselfParams, finite_sss_noself};
    use crate::graph::preds_of;

    #[test]
    fn test_finite_sss() {
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

        let data = finite_sss_noself(nnodes, &preds_of, &prb, &seeds, &NoselfParams { steps: 2 });
        println!("{:?}", &data.distr);

        assert_eq!(data.distr[0], 0.0);
        assert_eq!(data.distr[1], 1.0);
        assert_eq!(data.distr[2], 0.5);
        assert_eq!(data.distr[3], 0.25);
    }
}
