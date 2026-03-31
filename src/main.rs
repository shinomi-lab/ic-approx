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
    /// Seed number of PRNG.
    #[bpaf(short('r'), long("rng-seed"))]
    rng_seed: u64,
    /// Upper bound of uniform distr. of activation probability (included).
    #[bpaf(short('u'), long("upper"))]
    high_prob: f64,
    /// Number of seed nodes.
    #[bpaf(short('n'), long("nseeds"))]
    nseeds: usize,
    /// Default steps for finite methods
    #[bpaf(long("dsteps"))]
    default_steps: usize,
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
        rng_seed,
        high_prob,
        nseeds,
        default_steps,
        cmds,
    } = parser.run();
    let graph = read_edge_list(path, direction, separator)?;

    let mut body = ExecBody::new(graph, nseeds, high_prob, rng_seed, default_steps);
    for cmd in cmds {
        body.execute(cmd);
    }

    body.compare();

    // let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
    // let prb = generate_prob_mat(nnodes, &edges, high_prob, &mut rng);
    // let seeds = generate_seeds(nnodes, nseeds, &mut rng);
    // let preds_of = preds_of(nnodes, &edges);

    // let niter = 5000;
    // let adj = adj_binmat(nnodes, &edges);
    // let (mcl_distr, mcl_t, mcl_dur) = monte_carlo_ic_par(nnodes, &adj, &prb, &seeds, &mut rng, niter);
    // println!("Execution time (MCL): {} ms", mcl_dur.as_millis());

    // let t = mcl_t.ceil() as usize;
    // println!("{t}");

    // let (dmp_distr, dmp_dur) = finite_dmp(nnodes, &edges, &preds_of, &prb, &seeds, t);
    // println!("Execution time (DMP): {} ms", dmp_dur.as_millis());

    // let (tyr_distr, tyr_dur) =
    //     finite_taylor_scaled_point(nnodes, &preds_of, prb.transpose(), &seeds, t, 1.0);
    // println!("Execution time (TYR): {} ms", tyr_dur.as_millis());

    // let (tyr0_distr, tyr0_dur) =
    //     finite_taylor_zero_point(nnodes, &preds_of, prb.transpose(), &seeds, t);
    // println!("Execution time (TYR0): {} ms", tyr0_dur.as_millis());

    // let (sss_distr, sss_dur) = finite_sss_noself(nnodes, &preds_of, &prb, &seeds, t);
    // println!("Execution time (SSS): {} ms", sss_dur.as_millis());

    // println!(
    //     "Error (MCL vs DMP ): {:?}",
    //     error_of_distrs(&MCL_distr, &dmp_distr, nnodes)
    // );
    // println!(
    //     "Error (MCL vs TYR ): {:?}",
    //     error_of_distrs(&mcl_distr, &tyr_distr, nnodes)
    // );
    // println!(
    //     "Error (MCL vs TYR0): {:?}",
    //     error_of_distrs(&mcl_distr, &tyr0_distr, nnodes)
    // );
    // println!(
    //     "Error (MCL vs SSS): {:?}",
    //     error_of_distrs(&mcl_distr, &sss_distr, nnodes)
    // );
    // println!(
    //     "Error (SSS vs TYR ): {:?}",
    //     error_of_distrs(&sss_distr, &tyr_distr, nnodes)
    // );
    // println!(
    //     "Error (SSS vs TYR0): {:?}",
    //     error_of_distrs(&sss_distr, &tyr0_distr, nnodes)
    // );
    // println!(
    //     "Error (DMP vs TYR ): {:?}",
    //     error_of_distrs(&dmp_distr, &tyr_distr, nnodes)
    // );
    // println!(
    //     "Error (DMP vs TYR0): {:?}",
    //     error_of_distrs(&dmp_distr, &tyr0_distr, nnodes)
    // );
    // println!(
    //     "Error (TYR vs TYR0): {:?}",
    //     error_of_distrs(&tyr_distr, &tyr0_distr, nnodes)
    // );
    Ok(())
}
