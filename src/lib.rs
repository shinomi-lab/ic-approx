use std::collections::BTreeSet;
use std::error::Error;
use std::ops::Mul;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use std::{mem, time::Instant};

use bitvec::{bitvec, vec::BitVec};
use faer::traits::num_traits::Signed;
use faer::{Col, Mat, MatRef, unzip, zip};
use polars::error::PolarsResult;
use polars::prelude::{DataType, Field, LazyCsvReader, LazyFileListReader, PlRefPath, Schema};
use rand::distr::Uniform;
use rand::seq::index;
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

fn single_ic_model(
    nnodes: usize,
    adj: &[BitVec],
    prb: &Mat<f64>,
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
                if rng.random::<f64>() < prb[(i, j)] {
                    next_ca.set(j, true);
                }
            }
        }
        mem::swap(&mut ca, &mut next_ca);
        aa |= &next_ca;
    }
    (aa, t)
}

pub fn simulate_ic_model(
    nnodes: usize,
    adj: &[BitVec],
    prb: &Mat<f64>,
    seeds: &BitVec,
    rng: &mut impl Rng,
    niter: usize,
) -> (Col<f64>, f64, Duration) {
    let mut master_rng = Xoshiro256PlusPlus::from_rng(rng);
    let mut rngs = Vec::with_capacity(niter);

    rngs.push(master_rng.clone());
    for _ in 1..niter {
        master_rng.jump();
        rngs.push(master_rng.clone());
    }

    let results = rngs
        .par_iter_mut()
        .map(|rng| {
            let instant = Instant::now();
            let (aa, t) = single_ic_model(nnodes, adj, prb, seeds, rng);
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
    let m = niter as f64;
    v /= m;
    mean_t /= m;

    (v, mean_t, duration)
}

pub fn finite_dmp(
    nnodes: usize,
    edges: &[(usize, usize)],
    preds_of: &[Vec<usize>],
    prb: &Mat<f64>,
    seeds: &BitVec,
    t: usize,
) -> (Col<f64>, Duration) {
    let instant = Instant::now();

    let mut q_curr = Mat::<f64>::zeros(nnodes, nnodes);
    for &(i, j) in edges {
        q_curr[(i, j)] = f64::from(seeds[i]); // true -> 1.0, false -> 0.0
    }
    let mut q_next = Mat::<f64>::zeros(nnodes, nnodes);

    for _ in 0..t {
        for &(j, i) in edges {
            let temp = preds_of[j]
                .iter()
                .filter_map(|&l| {
                    if l == i {
                        None
                    } else {
                        Some(1.0 - prb[(l, j)] * q_curr[(l, j)])
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
            .map(|&j| 1.0 - prb[(j, i)] * q_curr[(j, i)])
            .fold(f64::from(!seeds[i]), f64::mul);
        1.0 - temp
    });

    (q, instant.elapsed())
}

fn pow2(&v: &f64) -> f64 {
    v * v
}

pub enum TaylorApprox {
    ScaledPoint {
        c1: Col<f64>,
        c2: Col<f64>,
        c3: Col<f64>,
    },
    ZeroPoint,
}

fn scaled_point(
    nnodes: usize,
    prb_t: MatRef<f64>,
    preds_of: &[Vec<usize>],
    scale: f64,
) -> TaylorApprox {
    let q = Col::<f64>::from_fn(nnodes, |i| {
        preds_of[i]
            .iter()
            .map(|&j| prb_t[(i, j)])
            .reduce(f64::min)
            .map(|p| p * scale)
            .unwrap_or(0.0)
    });
    // $\bar q^*$
    let r = q.map(|qi| 1.0 - qi);

    let indegs = Col::<u32>::from_fn(nnodes, |i| preds_of[i].len() as u32);
    let mut c0 = Col::<f64>::zeros(nnodes);
    zip!(&mut c0, &indegs, &q).for_each(|unzip!(c0i, &d, &qi)| {
        *c0i = qi * d as f64;
    });
    let c0 = c0;

    let mut c1 = Col::<f64>::zeros(nnodes);
    zip!(&mut c1, &r, &indegs).for_each(|unzip!(c1i, &ri, &d)| {
        *c1i = match (d, i32::try_from(d)) {
            (0, _) | (1, _) => 1.0,
            (_, Ok(n)) => ri.powi(n - 2),
            (_, Err(_)) => ri.powf(d as f64),
        };
    });
    let c1 = c1;

    let mut c2 = Col::<f64>::zeros(nnodes);
    zip!(&mut c2, &c0, &r, &indegs).for_each(|unzip!(c2i, &c0i, &ri, &d)| {
        *c2i = match d {
            0 | 1 => 1.0,
            _ => (c0i + ri).powi(2) - c0i,
        };
    });
    let c2 = c2;

    let mut c3 = Col::<f64>::zeros(nnodes);
    zip!(&mut c3, &c0, &q, &indegs).for_each(|unzip!(c3i, &c0i, &qi, &d)| {
        *c3i = match d {
            0 => 0.0,
            1 => 1.0,
            _ => 2.0 * c0i - 3.0 * qi + 1.0,
        };
    });
    let c3 = c3;

    TaylorApprox::ScaledPoint { c1, c2, c3 }
}

fn hadamard(v0: &Col<f64>, v1: &Col<f64>) -> Col<f64> {
    let mut w = Col::<f64>::zeros(v0.nrows());
    zip!(&mut w, v0, v1).for_each(|unzip!(w, v0i, v1i)| {
        *w = *v0i * v1i;
    });
    w
}

impl TaylorApprox {
    fn compute_bbar(&self, prb_t: MatRef<f64>, prb_t_sq: &Mat<f64>, y: &Col<f64>) -> Col<f64> {
        let yy = y.map(pow2);
        // let m = yy
        //     .iter()
        //     .enumerate()
        //     .filter_map(|(i, yi)| if yi.is_infinite() { Some(i) } else { None })
        //     .collect::<Vec<_>>();
        // println!("yy: {:?}", m);
        // for i in m {
        //     println!("{}", y[i]);
        // }
        let qy = prb_t * y;
        let temp = qy.map(pow2) - prb_t_sq * yy;
        match self {
            TaylorApprox::ScaledPoint { c1, c2, c3 } => {
                let mut temp = c2 - hadamard(c3, &qy) + temp;
                zip!(&mut temp, c1).for_each(|unzip!(tempi, &c1i)| {
                    let s = 1.0 - c1i * *tempi;
                    *tempi = if s.is_sign_negative() { 0.0 } else { s };
                });

                temp
            }
            TaylorApprox::ZeroPoint => {
                (&qy - temp / 2.0).map(|&v| if v.is_sign_negative() { 0.0 } else { v })
            }
        }
    }
}

/// `prob_t`: transposed probability matrix
pub fn finite_taylor_scaled_point(
    nnodes: usize,
    preds_of: &[Vec<usize>],
    prb_t: MatRef<f64>,
    seeds: &BitVec,
    t: usize,
    scale: f64,
) -> (Col<f64>, Duration) {
    let approx = scaled_point(nnodes, prb_t, preds_of, scale);
    finite_taylor(nnodes, prb_t, seeds, t, &approx)
}

pub fn finite_taylor_zero_point(
    nnodes: usize,
    prb_t: MatRef<f64>,
    seeds: &BitVec,
    t: usize,
) -> (Col<f64>, Duration) {
    finite_taylor(nnodes, prb_t, seeds, t, &TaylorApprox::ZeroPoint)
}

fn finite_taylor(
    nnodes: usize,
    prb_t: MatRef<f64>,
    seeds: &BitVec,
    t: usize,
    approx: &TaylorApprox,
) -> (Col<f64>, Duration) {
    let instant = Instant::now();
    let mut x = Col::<f64>::from_fn(nnodes, |i| f64::from(!seeds[i]));
    let mut y = Col::<f64>::from_fn(nnodes, |i| f64::from(seeds[i]));
    let mut z = Col::<f64>::zeros(nnodes);

    let prb_t_sq = prb_t.map(pow2);
    for _ in 0..t {
        let bb = approx.compute_bbar(prb_t, &prb_t_sq, &y);
        // let k = bb.iter().filter(|&&v| v <= -1.0).count();
        // println!("bb<0: {}", k);
        z += &y;
        zip!(&mut y, &x, &bb).for_each(|unzip!(yi, &xi, &bbi)| {
            *yi = xi * bbi;
        });
        zip!(&mut x, &y, &z).for_each(|unzip!(xi, &yi, &zi)| {
            *xi = 1.0 - yi - zi;
        });
        // let k = y.iter().filter(|&&yi| yi.abs() > 1.0).count();
        // println!("y>1: {}", k);
    }
    z += y;

    (z, instant.elapsed())
}

// map of predecessor list
pub fn preds_of(nnodes: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut preds_of = vec![Vec::<usize>::new(); nnodes];
    for &(i, j) in edges {
        preds_of[j].push(i);
    }
    preds_of
}

// list of indegrees
pub fn indegree_vec(nnodes: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut indegs = vec![0; nnodes];
    for &(_, j) in edges {
        indegs[j] += 1;
    }
    indegs
}

/// vector of rows
pub fn adj_binmat(nnodes: usize, edges: &[(usize, usize)]) -> Vec<BitVec> {
    let mut adj = vec![BitVec::repeat(false, nnodes); nnodes];
    for &(i, j) in edges {
        adj[i].set(j, true);
    }
    adj
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

pub fn generate_seeds(nnodes: usize, nseeds: usize, rng: &mut impl Rng) -> BitVec {
    let mut seeds = bitvec![0; nnodes];
    for i in index::sample(rng, nnodes, nseeds) {
        seeds.set(i, true);
    }
    seeds
}

pub fn count_nnodes(edges: &[(usize, usize)]) -> usize {
    let mut set = BTreeSet::new();
    let mut nnodes = 0;
    for &(i, j) in edges {
        if set.insert(i) {
            nnodes += 1;
        }
        if set.insert(j) {
            nnodes += 1;
        }
    }
    nnodes
}

const SOURCE: &'static str = "source";
const TARGET: &'static str = "target";

#[derive(Debug)]
pub enum Direction {
    Directed,
    Undirected,
}

impl FromStr for Direction {
    type Err = Box<dyn Error>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "directed" => Ok(Direction::Directed),
            "undirected" => Ok(Direction::Undirected),
            _ => Err(format!("Cannot parse \"{s}\" into direction.").into()),
        }
    }
}

pub struct GraphData {
    pub nnodes: usize,
    pub edges: Vec<(usize, usize)>,
    pub direction: Direction,
}

pub fn read_edge_list<P: AsRef<Path>>(
    path: P,
    direction: Direction,
    separator: u8,
    has_header: bool,
) -> PolarsResult<GraphData> {
    use polars::prelude as pl;

    let lf = LazyCsvReader::new(PlRefPath::try_from_path(path.as_ref())?)
        .with_has_header(has_header)
        .with_separator(separator)
        .with_schema(Some(Arc::new(Schema::from_iter(vec![
            Field::new(SOURCE.into(), DataType::UInt32),
            Field::new(TARGET.into(), DataType::UInt32),
        ]))))
        .finish()?;

    let min_max_df = lf
        .clone()
        .select([pl::concat_list([SOURCE, TARGET])?
            .explode(pl::ExplodeOptions {
                empty_as_null: false,
                keep_nulls: false,
            })
            .alias("ST")])
        .select([
            pl::col("ST").min().alias("min"),
            pl::col("ST").max().alias("max"),
        ])
        .collect()?;

    let [min_col, max_col] = min_max_df.columns().as_array().unwrap();
    let min = min_col.u32()?.first().unwrap();
    let max = max_col.u32()?.first().unwrap();

    let df = lf.collect()?;
    let source_iter = df.column(SOURCE)?.u32()?;
    let target_iter = df.column(TARGET)?.u32()?;

    let iter = source_iter.iter().zip(target_iter).filter_map(|st| {
        if let (Some(i), Some(j)) = st {
            Some(((i - min) as usize, (j - min) as usize))
        } else {
            None
        }
    });

    let edges = match direction {
        Direction::Directed => iter.collect(),
        Direction::Undirected => iter.flat_map(|(i, j)| [(i, j), (j, i)]).collect(),
    };

    Ok(GraphData {
        nnodes: (max - min + 1) as usize,
        edges,
        direction,
    })
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
    use std::collections::BTreeSet;

    use bitvec::{bits, bitvec, prelude::Lsb0, vec::BitVec};
    use faer::{Col, Mat};
    use rand::{SeedableRng, rngs::SmallRng};

    use crate::{
        Direction, GraphData, adj_binmat, finite_dmp, finite_taylor_scaled_point,
        generate_prob_mat, generate_seeds, preds_of, read_edge_list, single_ic_model,
    };

    #[test]
    fn test_read_edge_list() {
        let graph = read_edge_list(
            "./test/graph.txt",
            crate::Direction::Undirected,
            b' ',
            false,
        )
        .unwrap();
        assert_eq!(graph.nnodes, 4);
        let edges = BTreeSet::from_iter(graph.edges);
        assert_eq!(
            edges,
            BTreeSet::from_iter([(0, 1), (0, 2), (1, 3), (1, 0), (2, 0), (3, 1)])
        );
        assert!(matches!(graph.direction, Direction::Undirected));

        let graph = read_edge_list("./test/graph.txt", Direction::Directed, b' ', false).unwrap();
        assert_eq!(graph.nnodes, 4);
        let edges = BTreeSet::from_iter(graph.edges);
        assert_eq!(edges, BTreeSet::from_iter([(0, 1), (0, 2), (1, 3)]));
        assert!(matches!(graph.direction, Direction::Directed));
    }

    #[test]
    fn test_adj_bitmat() {
        let nnodes = 5;
        let edges = vec![(0, 1), (1, 0), (2, 3), (4, 3)];

        let adj = adj_binmat(nnodes, &edges);
        assert_eq!(adj.len(), nnodes);

        let edges = BTreeSet::from_iter(edges);
        for (i, row) in adj.iter().enumerate() {
            assert_eq!(row.len(), nnodes);
            for j in 0..nnodes {
                if edges.contains(&(i, j)) {
                    assert!(row[j]);
                } else {
                    assert!(!row[j]);
                }
            }
        }
    }

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

    #[test]
    fn test_preds_of() {
        let nnodes = 4;
        let edges = vec![(0, 1), (0, 2), (1, 2), (1, 3)];
        let preds_of = preds_of(nnodes, &edges);
        assert_eq!(preds_of[0], Vec::<usize>::new());
        assert_eq!(preds_of[1], vec![0]);
        assert_eq!(preds_of[2], vec![0, 1]);
        assert_eq!(preds_of[3], vec![1]);
    }

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
        let (res, _) = single_ic_model(nnodes, &adj, &prb, &seeds, &mut rng);
        assert!(!res[0]);
        assert!(res[1]);
        assert!(res[2]);
        assert!(res[3]);
    }

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

        let (res, _) = finite_dmp(nnodes, &edges, &preds_of, &prb, &seeds, 2);

        assert_eq!(res[0], 0.0);
        assert_eq!(res[1], 1.0);
        assert_eq!(res[2], 0.5);
        assert_eq!(res[3], 0.25);
    }

    #[test]
    fn test_finite_taylor() {
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

        let (res, _) =
            finite_taylor_scaled_point(nnodes, &preds_of, prb.transpose(), &seeds, 2, 1.0);
        println!("{:?}", &res);

        assert_eq!(res[0], 0.0);
        assert_eq!(res[1], 1.0);
        assert_eq!(res[2], 0.5);
        assert_eq!(res[3], 0.25);
    }

    #[test]
    fn test_taylor_bug() {
        let GraphData { nnodes, edges, .. } = read_edge_list(
            "/Users/masaaki/Downloads/facebook_combined.txt.gz",
            crate::Direction::Undirected,
            b' ',
            false,
        )
        .unwrap();
        assert_eq!(nnodes, 4039);
        assert_eq!(edges.len(), 88_234 * 2);

        let prb = {
            let mut prb = Mat::<f64>::zeros(nnodes, nnodes);
            for &e in &edges {
                prb[e] = 0.5;
            }
            prb
        };
        let mut seeds = bitvec![0; nnodes];
        for i in 0..10 {
            seeds.set(i, true);
        }
        let preds_of = preds_of(nnodes, &edges);

        let t = 15;

        let (res, _) =
            finite_taylor_scaled_point(nnodes, &preds_of, prb.transpose(), &seeds, t, 0.5);
        let nans = res
            .iter()
            .enumerate()
            .filter_map(|(i, r)| if r.is_nan() { Some(i) } else { None })
            .collect::<Vec<_>>();
        dbg!(nans.len());
    }

    #[test]
    fn test_mat() {
        let n = 4040;
        let mut a = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = 0.5;
            }
        }
        let v = Col::<f64>::zeros(n);
        let w = a * v;
        dbg!(w);
    }
}
