pub mod dmp;
pub mod graph;
pub mod ic_model;
pub mod sss;
pub mod taylor;

use std::{collections::BTreeMap, time::Duration};

use bitvec::{bitvec, vec::BitVec};
use faer::{Col, Mat, traits::num_traits::Signed};
use rand::{Rng, RngExt, SeedableRng, distr::Uniform, seq::index};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::graph::Graph;

pub fn generate_seeds(nnodes: usize, nseeds: usize, rng: &mut impl Rng) -> BitVec {
    let mut seeds = bitvec![0; nnodes];
    for i in index::sample(rng, nnodes, nseeds) {
        seeds.set(i, true);
    }
    seeds
}

pub fn generate_prob_mat(
    nnodes: usize,
    edges: &[(usize, usize)],
    high: f64,
    rng: &mut impl Rng,
) -> Mat<f64> {
    let mut prb = Mat::<f64>::zeros(nnodes, nnodes);
    for &(i, j) in edges {
        let p = 1.0 - rng.sample(Uniform::new(1.0 - high, 1.0).unwrap());
        prb[(i, j)] = p;
    }
    prb
}

pub struct InfluenceData {
    distr: Col<f64>,
    dur: Duration,
}

impl InfluenceData {
    fn new(distr: Col<f64>, dur: Duration) -> Self {
        Self { distr, dur }
    }

    fn duration(&self) -> u128 {
        self.dur.as_millis()
    }
}

#[allow(dead_code)]
pub struct InfluenceParams {
    seeds: BitVec,
    prob: Mat<f64>,
    graph: Graph,
    rng_seed: u64,
    default_steps: usize,
}

#[derive(Debug, Clone, bpaf::Bpaf)]
pub enum AlgMode {
    /// Finite loops with default steps
    #[bpaf(long("fn"))]
    Finite,
    /// Infinite loops
    #[bpaf(long("inf"))]
    Infinite,
}

#[derive(Debug, Clone, bpaf::Bpaf)]
pub enum ExecCmd {
    #[bpaf(command("mcm"), adjacent)]
    /// Monte-Carlo method
    MonteCarlo {
        #[bpaf(short, long("iter"))]
        /// Number of iteration
        iter_size: usize,
        #[bpaf(long("mr"))]
        /// Seed value of PRNG
        rng_seed: u64,
    },
    #[bpaf(command("dmp"), adjacent)]
    /// Dynamic message passing
    DMP {
        #[bpaf(external(alg_mode))]
        mode: AlgMode,
    },
    #[bpaf(command("sssn"), adjacent)]
    /// SSS-Noself
    SSSNoself {
        #[bpaf(external(alg_mode))]
        mode: AlgMode,
    },
    #[bpaf(command("tyl"), adjacent)]
    /// 2nd Taylor approximation
    Taylor {
        /// Scale of $q^\ast_i$ (default: 1.0)
        #[bpaf(long, fallback(1.0))]
        scale: f64,
        #[bpaf(external(alg_mode))]
        mode: AlgMode,
    },
    #[bpaf(command("tyl0"), adjacent)]
    /// 2nd Taylor approximation at zero point
    TaylorZero {
        #[bpaf(external(alg_mode))]
        mode: AlgMode,
    },
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum ExecKey {
    MonteCarlo,
    DMP,
    SSSNoself,
    Taylor,
    TaylorZero,
}

impl ExecKey {
    fn long_name(&self) -> &str {
        match self {
            ExecKey::MonteCarlo => "Monte-Carlo",
            ExecKey::DMP => "Dynamic message passing",
            ExecKey::SSSNoself => "SSS-Noself",
            ExecKey::Taylor => "2nd Taylor",
            ExecKey::TaylorZero => "2nd Taylor at zero point",
        }
    }
    fn short_name(&self) -> &str {
        match self {
            ExecKey::MonteCarlo => "MCL",
            ExecKey::DMP => "DMP",
            ExecKey::SSSNoself => "SSSN",
            ExecKey::Taylor => "TYL",
            ExecKey::TaylorZero => "TYL0",
        }
    }
}

#[derive(Default)]
struct ExecResult {
    estimated_steps: Option<usize>,
    table: BTreeMap<ExecKey, InfluenceData>,
}

pub struct ExecBody {
    params: InfluenceParams,
    result: ExecResult,
}

impl ExecBody {
    pub fn new(
        graph: Graph,
        nseeds: usize,
        high_prob: f64,
        rng_seed: u64,
        default_steps: usize,
    ) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
        let prob = generate_prob_mat(graph.nnodes(), graph.edges(), high_prob, &mut rng);
        let seeds = generate_seeds(graph.nnodes(), nseeds, &mut rng);
        Self {
            params: InfluenceParams {
                seeds,
                prob,
                graph,
                rng_seed,
                default_steps,
            },
            result: ExecResult::default(),
        }
    }

    pub fn execute(&mut self, cmd: ExecCmd) {
        let key = cmd.key();
        println!("start {}", key.long_name());
        let data = match cmd {
            ExecCmd::MonteCarlo {
                iter_size,
                rng_seed,
            } => {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
                let (data, extra) = ic_model::monte_carlo_ic_par(
                    self.params.graph.nnodes(),
                    self.params.graph.adj(),
                    &self.params.prob,
                    &self.params.seeds,
                    &mut rng,
                    iter_size,
                );
                self.result.estimated_steps = Some(extra.mean_steps.ceil() as usize);
                println!("{extra:?}");
                data
            }
            ExecCmd::DMP {
                mode: AlgMode::Finite,
            } => {
                let params = dmp::Params {
                    steps: self
                        .result
                        .estimated_steps
                        .unwrap_or(self.params.default_steps),
                };
                let data = dmp::finite_dmp(
                    self.params.graph.nnodes(),
                    self.params.graph.edges(),
                    self.params.graph.preds_of(),
                    &self.params.prob,
                    &self.params.seeds,
                    &params,
                );
                println!("{params:?}");
                data
            }
            ExecCmd::SSSNoself {
                mode: AlgMode::Finite,
            } => {
                let params = sss::NoselfParams {
                    steps: self
                        .result
                        .estimated_steps
                        .unwrap_or(self.params.default_steps),
                };
                let data = sss::finite_sss_noself(
                    self.params.graph.nnodes(),
                    self.params.graph.preds_of(),
                    &self.params.prob,
                    &self.params.seeds,
                    &params,
                );
                println!("{params:?}");
                data
            }
            ExecCmd::Taylor {
                scale,
                mode: AlgMode::Finite,
            } => {
                let params = taylor::ScaledPointParams {
                    steps: self
                        .result
                        .estimated_steps
                        .unwrap_or(self.params.default_steps),
                    scale,
                };
                let data = taylor::finite_taylor_scaled_point(
                    self.params.graph.nnodes(),
                    self.params.graph.indegs(),
                    self.params.graph.preds_of(),
                    self.params.prob.transpose(),
                    &self.params.seeds,
                    &params,
                );
                println!("{params:?}");
                data
            }
            ExecCmd::TaylorZero {
                mode: AlgMode::Finite,
            } => {
                let params = taylor::ZeroPointParams {
                    steps: self
                        .result
                        .estimated_steps
                        .unwrap_or(self.params.default_steps),
                };
                let data = taylor::finite_taylor_zero_point(
                    self.params.graph.nnodes(),
                    self.params.graph.preds_of(),
                    self.params.prob.transpose(),
                    &self.params.seeds,
                    &params,
                );
                println!("{params:?}");
                data
            }
            _ => todo!(),
        };
        let entry = self.result.table.entry(key);
        let d = entry.insert_entry(data).get().duration();
        println!("finished in {} ms", d);
    }

    pub fn compare(&self) {
        let keys = self.result.table.keys().collect::<Vec<_>>();
        for (i, &k1) in keys.iter().enumerate() {
            for &k2 in keys.iter().skip(i + 1) {
                let stat = error_of_distrs(
                    &self.result.table[k1].distr,
                    &self.result.table[k2].distr,
                    self.params.graph.nnodes(),
                );
                println!(
                    "Error ({} vs {}): {:?}",
                    k1.short_name(),
                    k2.short_name(),
                    stat,
                );
            }
        }
    }
}

impl ExecCmd {
    fn key(&self) -> ExecKey {
        match self {
            ExecCmd::MonteCarlo { .. } => ExecKey::MonteCarlo,
            ExecCmd::DMP { .. } => ExecKey::DMP,
            ExecCmd::SSSNoself { .. } => ExecKey::SSSNoself,
            ExecCmd::Taylor { .. } => ExecKey::Taylor,
            ExecCmd::TaylorZero { .. } => ExecKey::TaylorZero,
        }
    }
}

#[derive(Debug)]
pub struct Stat {
    pub mean: f64,
    pub var: f64,
}

pub fn error_of_distrs(d0: &Col<f64>, d1: &Col<f64>, n: usize) -> Stat {
    let n = n as f64;
    let errs = (d0 - d1).map(|v| v.abs());
    let mean = errs.sum() / n;
    let var = errs.map(|&e| (e - mean).powi(2)).sum() / n;
    Stat { mean, var }
}

#[cfg(test)]
mod tests {
    use super::generate_seeds;
    use crate::generate_prob_mat;
    use rand::{SeedableRng, rngs::SmallRng};
    use std::collections::BTreeSet;

    #[test]
    fn test_generate_seeds() {
        let mut rng = SmallRng::seed_from_u64(0);
        let seeds = generate_seeds(4, 2, &mut rng);
        assert_eq!(seeds.len(), 4);
        assert_eq!(seeds.iter_ones().len(), 2);
    }

    #[test]
    fn test_generate_prob_mat() {
        let nnodes = 5;
        let edges = vec![(0, 1), (1, 0), (2, 3), (4, 3)];

        let mut rng = SmallRng::seed_from_u64(0);
        let prb = generate_prob_mat(nnodes, &edges, 0.5, &mut rng);

        assert_eq!(prb.shape(), (nnodes, nnodes));

        let edges = BTreeSet::from_iter(edges);
        for i in 0..nnodes {
            for j in 0..nnodes {
                let e = (i, j);
                let p = prb[e];
                if edges.contains(&e) {
                    assert!(p > 0.0 && p <= 0.5);
                } else {
                    assert!(p == 0.0);
                }
            }
        }
    }
}
