use std::time::Instant;

use bitvec::vec::BitVec;
use faer::{Col, Mat, MatRef};

use crate::InfluenceData;

pub enum TaylorApprox {
    ScaledPoint {
        c1: Col<f64>,
        c2: Col<f64>,
        c3: Col<f64>,
    },
    ZeroPoint {
        c3: Col<bool>,
    },
}

fn pow2(&v: &f64) -> f64 {
    v * v
}

fn scaled_point(
    nnodes: usize,
    indegs: &Col<u32>,
    prob_t: MatRef<f64>,
    preds_of: &[Vec<usize>],
    scale: f64,
) -> TaylorApprox {
    let q = Col::<f64>::from_fn(nnodes, |i| {
        preds_of[i]
            .iter()
            .map(|&j| prob_t[(i, j)])
            .reduce(f64::min)
            .map(|p| p * scale)
            .unwrap_or(0.0)
    });
    let c1 = Col::<f64>::from_fn(nnodes, |i| match indegs[i] {
        0 | 1 => 1.0,
        d => {
            let r = 1.0 - q[i];
            i32::try_from(d).map_or_else(|_| r.powf(d as f64), |n| r.powi(n - 2))
        }
    });
    let c2 = Col::<f64>::from_fn(nnodes, |i| match indegs[i] {
        0 | 1 => 0.0,
        d => ((d - 2) as f64 * q[i]) * ((d - 1) as f64 * q[i] + 2.0),
    });
    let c3 = Col::<f64>::from_fn(nnodes, |i| match indegs[i] {
        0 | 1 => 1.0,
        d => (d - 2) as f64 * q[i] + 1.0,
    });

    TaylorApprox::ScaledPoint { c1, c2, c3 }
}

fn zero_point(nnodes: usize, preds_of: &[Vec<usize>]) -> TaylorApprox {
    let c3 = Col::<bool>::from_fn(nnodes, |i| !preds_of[i].is_empty());
    TaylorApprox::ZeroPoint { c3 }
}

impl TaylorApprox {
    fn compute_bbar(
        &self,
        nnodes: usize,
        prob_t: MatRef<f64>,
        prob_t_sq: &Mat<f64>,
        y: &Col<f64>,
    ) -> Col<f64> {
        let yy = y.map(pow2);
        let qy = prob_t * y;
        let temp = (qy.map(pow2) - prob_t_sq * yy) / 2.0;
        match self {
            TaylorApprox::ScaledPoint { c1, c2, c3 } => Col::from_fn(nnodes, |i| {
                let s = 1.0 - c1[i] * (1.0 + c2[i] / 2.0 - c3[i] * qy[i] + temp[i]);
                if s.is_sign_negative() { 0.0 } else { s }
            }),
            TaylorApprox::ZeroPoint { c3 } => Col::from_fn(nnodes, |i| {
                let s = if c3[i] { qy[i] } else { 0.0 } - temp[i];
                if s.is_sign_negative() { 0.0 } else { s }
            }),
        }
    }
}

#[derive(Debug)]
pub struct ScaledPointParams {
    pub steps: usize,
    pub scale: f64,
}

#[derive(Debug)]
pub struct ZeroPointParams {
    pub steps: usize,
}

/// `prob_t`: transposed probability matrix
pub fn finite_taylor_scaled_point(
    nnodes: usize,
    indegs: &Col<u32>,
    preds_of: &[Vec<usize>],
    prob_t: MatRef<f64>,
    seeds: &BitVec,
    params: &ScaledPointParams,
) -> InfluenceData {
    let instant = Instant::now();
    let approx = scaled_point(nnodes, indegs, prob_t, preds_of, params.scale);
    let distr = finite_taylor(nnodes, prob_t, seeds, params.steps, &approx);
    InfluenceData::new(distr, instant.elapsed())
}

pub fn finite_taylor_zero_point(
    nnodes: usize,
    preds_of: &[Vec<usize>],
    prob_t: MatRef<f64>,
    seeds: &BitVec,
    params: &ZeroPointParams,
) -> InfluenceData {
    let instant = Instant::now();
    let approx = zero_point(nnodes, preds_of);
    let distr = finite_taylor(nnodes, prob_t, seeds, params.steps, &approx);
    InfluenceData::new(distr, instant.elapsed())
}

fn finite_taylor(
    nnodes: usize,
    prob_t: MatRef<f64>,
    seeds: &BitVec,
    t: usize,
    approx: &TaylorApprox,
) -> Col<f64> {
    // not activated yet
    let mut x = Col::<f64>::from_fn(nnodes, |i| f64::from(!seeds[i]));
    // currently activated
    let mut y = Col::<f64>::from_fn(nnodes, |i| f64::from(seeds[i]));
    // previously activated
    let mut z = Col::<f64>::zeros(nnodes);

    let prob_t_sq = prob_t.map(pow2);
    for _ in 0..t {
        let bb = approx.compute_bbar(nnodes, prob_t, &prob_t_sq, &y);
        for i in 0..nnodes {
            z[i] += y[i];
            let yi = x[i] * bb[i];
            y[i] = yi;
            x[i] = 1.0 - yi - z[i];
        }
    }
    z + y
}

#[cfg(test)]
mod tests {
    use super::{ScaledPointParams, finite_taylor_scaled_point};
    use crate::graph::{indegs, preds_of};

    use bitvec::{bits, prelude::Lsb0, vec::BitVec};
    use faer::Mat;

    #[test]
    fn test_finite_taylor() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let nnodes = 4;
        let prob = {
            let mut prob = Mat::<f64>::zeros(nnodes, nnodes);
            for &e in &edges {
                prob[e] = 0.5;
            }
            prob
        };
        let seeds = BitVec::from_bitslice(&bits![0, 1, 0, 0]);
        let indegs = indegs(nnodes, &edges);
        let preds_of = preds_of(nnodes, &edges);

        let data = finite_taylor_scaled_point(
            nnodes,
            &indegs,
            &preds_of,
            prob.transpose(),
            &seeds,
            &ScaledPointParams {
                steps: 2,
                scale: 1.0,
            },
        );
        println!("{:?}", &data.distr);

        assert_eq!(data.distr[0], 0.0);
        assert_eq!(data.distr[1], 1.0);
        assert_eq!(data.distr[2], 0.5);
        assert_eq!(data.distr[3], 0.25);
    }
}
