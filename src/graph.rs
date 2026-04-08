use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::OsStr;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;
use std::{fs, io, u32};

use anyhow::anyhow;
use bitvec::vec::BitVec;
use faer::Col;
use flate2::read::MultiGzDecoder;
use rayon::prelude::*;

// map of predecessor list
pub fn preds_of(nnodes: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut preds_of = vec![Vec::<usize>::new(); nnodes];
    for &(i, j) in edges {
        preds_of[j].push(i);
    }
    preds_of
}

// indegree vector
pub fn indegs(nnodes: usize, edges: &[(usize, usize)]) -> Col<u32> {
    let mut indegs = Col::from_fn(nnodes, |_| 0);
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

#[allow(dead_code)]
pub struct Graph {
    pub nnodes: usize,
    pub edges: Vec<(usize, usize)>,
    pub direction: Direction,
    pub adj: Vec<BitVec>,
    pub preds_of: Vec<Vec<usize>>,
    pub indegs: Col<u32>,
}

impl Graph {
    pub fn new(nnodes: usize, edges: Vec<(usize, usize)>, direction: Direction) -> Self {
        let adj = adj_binmat(nnodes, &edges);
        let preds_of = preds_of(nnodes, &edges);
        let indegs = indegs(nnodes, &edges);
        Self {
            nnodes,
            edges,
            direction,
            adj,
            preds_of,
            indegs,
        }
    }
}

fn parse_reader<R: io::Read + Send>(
    reader: BufReader<R>,
    separator: char,
) -> anyhow::Result<Vec<(u32, u32)>> {
    let pairs = reader
        .lines()
        .par_bridge()
        .map(|res| {
            let line = res?;
            let mut iter = line.split(separator);
            let i = iter.next().ok_or(anyhow!("invalid data"))?.parse::<u32>()?;
            let j = iter.next().ok_or(anyhow!("invalid data"))?.parse::<u32>()?;
            Ok((i, j))
        })
        .collect::<Result<Vec<_>, anyhow::Error>>()?;
    Ok(pairs)
}

pub fn read_edge_list<P: AsRef<Path>>(
    path: P,
    direction: Direction,
    separator: char,
) -> anyhow::Result<Graph> {
    let file = fs::File::open(&path)?;
    let pairs = match path.as_ref().extension().map(OsStr::to_str).flatten() {
        Some("gz") => parse_reader(io::BufReader::new(MultiGzDecoder::new(file)), separator)?,
        _ => parse_reader(io::BufReader::new(file), separator)?,
    };

    let mut min = u32::MAX;
    let mut max = u32::MIN;

    let mut table = BTreeMap::<u32, BTreeSet<u32>>::new();
    for (from, to) in pairs {
        min = from.min(to).min(min);
        max = from.max(to).max(max);
        // skip self-loop edges
        if from == to {
            continue;
        }
        table.entry(from).or_default().insert(to);
    }

    let iter = table.into_iter().flat_map(|(from, tos)| {
        tos.into_iter()
            .map(|to| ((from - min) as usize, (to - min) as usize))
            .collect::<Vec<_>>()
    });

    let edges = match direction {
        Direction::Directed => iter.collect(),
        Direction::Undirected => iter.flat_map(|(i, j)| [(i, j), (j, i)]).collect(),
    };

    Ok(Graph::new((max - min + 1) as usize, edges, direction))
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};
    use std::collections::BTreeSet;

    use super::{Direction, adj_binmat, preds_of, read_edge_list};
    use crate::generate_seeds;

    #[test]
    fn test_read_edge_list() {
        let graph = read_edge_list("./test/graph.txt", Direction::Undirected, ' ').unwrap();
        assert_eq!(graph.nnodes, 4);
        let edges = BTreeSet::from_iter(graph.edges);
        assert_eq!(
            edges,
            BTreeSet::from_iter([(0, 1), (0, 2), (1, 3), (1, 0), (2, 0), (3, 1)])
        );
        assert!(matches!(graph.direction, Direction::Undirected));

        let graph = read_edge_list("./test/graph.txt", Direction::Directed, ' ').unwrap();
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
    fn test_preds_of() {
        let nnodes = 4;
        let edges = vec![(0, 1), (0, 2), (1, 2), (1, 3)];
        let preds_of = preds_of(nnodes, &edges);
        assert_eq!(preds_of[0], Vec::<usize>::new());
        assert_eq!(preds_of[1], vec![0]);
        assert_eq!(preds_of[2], vec![0, 1]);
        assert_eq!(preds_of[3], vec![1]);
    }
}
