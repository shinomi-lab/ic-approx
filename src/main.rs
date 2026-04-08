use std::path::PathBuf;

use ic_approx::{graph::*, *};

#[derive(bpaf::Bpaf)]
#[bpaf(options)]
struct Options {
    /// File path of a graph edge list.
    #[bpaf(short('g'), long("graph-path"))]
    path: PathBuf,
    /// Direction of a graph. (directed/undirected)
    #[bpaf(short('d'), long("direction"))]
    direction: Direction,
    /// A character to separate columns of an edge list. (default: space)
    #[bpaf(long("sep"), fallback(' '))]
    separator: char,
    /// Seed value of PRNG for initial acivated nodes.
    #[bpaf(long("ns"))]
    rng_seed_in: u64,
    /// Seed value of PRNG for distr. of activation prob..
    #[bpaf(long("ps"))]
    rng_seed_ap: u64,
    /// Upper bound of uniform distr. of activation prob. (included).
    #[bpaf(short('u'), long("upper"))]
    high_prob: f64,
    /// Number of seed nodes.
    #[bpaf(short('n'), long("nseeds"))]
    nseeds: usize,
    /// Default steps for finite methods
    #[bpaf(long("dsteps"))]
    default_steps: usize,
    /// Number of times to repeat methods except Monte-Carlo
    #[bpaf(long("rpt"))]
    nrepts: usize,
    /// File path to output computation time.
    #[bpaf(long("tot"))]
    time_path: PathBuf,
    /// File path to output errors.
    #[bpaf(long("toe"))]
    error_path: PathBuf,
    #[bpaf(external(exec_cmd), many)]
    cmds: Vec<ExecCmd>,
}

fn main() -> anyhow::Result<()> {
    let parser = options()
        .version("0.1.0")
        .descr("Experiments of approximate methods for IC model");

    let Options {
        path,
        direction,
        separator,
        rng_seed_in,
        rng_seed_ap,
        high_prob,
        nseeds,
        default_steps,
        nrepts,
        time_path,
        error_path,
        cmds,
    } = parser.run();
    let graph = read_edge_list(path, direction, separator)?;

    let body = ExecBody::new(
        graph,
        nseeds,
        high_prob,
        rng_seed_in,
        rng_seed_ap,
        default_steps,
        nrepts,
    );
    let result = body.execute_all(cmds);
    // result.compare();
    result.write(time_path, error_path)?;

    Ok(())
}
