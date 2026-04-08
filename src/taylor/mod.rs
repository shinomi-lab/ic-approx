use std::time::Instant;

use bitvec::vec::BitVec;
use faer::{Col, Mat, MatRef};

use crate::{InfluenceData, graph::Graph};

#[allow(dead_code)]
struct TaylorApproxBuilder<'a> {
    graph: &'a Graph,
    prob_t: MatRef<'a, f64>,
    t: usize,
    approx: TaylorApprox,
}

enum TaylorApprox {
    ScaledPoint {
        b1: Col<f64>,
        b2: Col<f64>,
        b3: Col<f64>,
    },
    ZeroPoint,
}

fn pow2(&v: &f64) -> f64 {
    v * v
}

impl<'a> TaylorApproxBuilder<'a> {
    fn scaled_point(graph: &'a Graph, prob_t: MatRef<'a, f64>, params: &ScaledPointParams) -> Self {
        let Graph {
            nnodes,
            preds_of,
            indegs,
            ..
        } = graph;
        let nnodes = *nnodes;
        let q = Col::<f64>::from_fn(nnodes, |i| {
            preds_of[i]
                .iter()
                .map(|&j| prob_t[(i, j)])
                .reduce(f64::min)
                .map(|p| p * params.scale)
                .unwrap_or(0.0)
        });
        let b1 = Col::<f64>::from_fn(nnodes, |i| match indegs[i] {
            0 | 1 => 1.0,
            d => {
                let r = 1.0 - q[i];
                i32::try_from(d).map_or_else(|_| r.powf(d as f64), |n| r.powi(n - 2))
            }
        });
        let b2 = Col::<f64>::from_fn(nnodes, |i| match indegs[i] {
            0 | 1 => 0.0,
            d => ((d - 2) as f64 * q[i]) * ((d - 1) as f64 * q[i] + 2.0),
        });
        let b3 = Col::<f64>::from_fn(nnodes, |i| match indegs[i] {
            0 | 1 => 1.0,
            d => (d - 2) as f64 * q[i] + 1.0,
        });

        Self {
            graph,
            prob_t,
            t: params.steps,
            approx: TaylorApprox::ScaledPoint { b1, b2, b3 },
        }
    }

    fn zero_point(graph: &'a Graph, prob_t: MatRef<'a, f64>, params: &ZeroPointParams) -> Self {
        Self {
            graph,
            prob_t,
            t: params.steps,
            approx: TaylorApprox::ZeroPoint,
        }
    }

    fn finite_taylor(self, seeds: &BitVec) -> Col<f64> {
        let nnodes = self.graph.nnodes;
        // not activated yet
        let mut x = Col::<f64>::from_fn(nnodes, |i| f64::from(!seeds[i]));
        // currently activated
        let mut y = Col::<f64>::from_fn(nnodes, |i| f64::from(seeds[i]));
        // previously activated
        let mut z = Col::<f64>::zeros(nnodes);

        let prob_t_sq = self.prob_t.map(pow2);
        for _ in 0..self.t {
            let ap = self.compute_approximate_value(&prob_t_sq, &y);
            for i in 0..nnodes {
                z[i] += y[i];
                let yi = x[i] * ap[i];
                // let yi = if yi.is_sign_negative() {
                //     0.0
                // } else if yi > 1.0 {
                //     1.0
                // } else {
                //     yi
                // };
                y[i] = yi;
                x[i] = 1.0 - yi - z[i];
            }
        }
        z + y
    }

    fn compute_approximate_value(&self, prob_t_sq: &Mat<f64>, y: &Col<f64>) -> Col<f64> {
        let yy = y.map(pow2);
        let qy = self.prob_t * y;
        let temp = (qy.map(pow2) - prob_t_sq * yy) / 2.0;
        match &self.approx {
            TaylorApprox::ScaledPoint { b1, b2, b3 } => Col::from_fn(self.graph.nnodes, |i| {
                let ap = b1[i] * (1.0 + b2[i] / 2.0 - b3[i] * qy[i] + temp[i]);
                let ap = if ap.is_sign_negative() {
                    0.0
                } else if ap > 1.0 {
                    1.0
                } else {
                    ap
                };
                1.0 - ap
            }),
            TaylorApprox::ZeroPoint => Col::from_fn(self.graph.nnodes, |i| {
                if self.graph.indegs[i] > 0 {
                    let ap = qy[i] - temp[i];
                    let ap = if ap.is_sign_negative() {
                        0.0
                    } else if ap > 1.0 {
                        1.0
                    } else {
                        ap
                    };
                    ap
                } else {
                    1.0
                }
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
    graph: &Graph,
    prob_t: MatRef<f64>,
    seeds: &BitVec,
    params: &ScaledPointParams,
) -> InfluenceData {
    let instant = Instant::now();
    let build = TaylorApproxBuilder::scaled_point(graph, prob_t, params);
    let distr = build.finite_taylor(seeds);
    InfluenceData::new(distr, instant.elapsed())
}

pub fn finite_taylor_zero_point(
    graph: &Graph,
    prob_t: MatRef<f64>,
    seeds: &BitVec,
    params: &ZeroPointParams,
) -> InfluenceData {
    let instant = Instant::now();
    let build = TaylorApproxBuilder::zero_point(graph, prob_t, params);
    let distr = build.finite_taylor(seeds);
    InfluenceData::new(distr, instant.elapsed())
}

#[cfg(test)]
mod tests {
    use super::{ScaledPointParams, finite_taylor_scaled_point};
    use crate::graph::Graph;

    use bitvec::{bits, prelude::Lsb0, vec::BitVec};
    use faer::Mat;

    #[test]
    fn test_finite_taylor() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let nnodes = 4;
        let graph = Graph::new(nnodes, edges, crate::graph::Direction::Directed);
        let prob = {
            let mut prob = Mat::<f64>::zeros(nnodes, nnodes);
            for &e in &graph.edges {
                prob[e] = 0.5;
            }
            prob
        };
        let seeds = BitVec::from_bitslice(&bits![0, 1, 0, 0]);

        let data = finite_taylor_scaled_point(
            &graph,
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
