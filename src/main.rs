use std::path::PathBuf;

use bpaf::Parser;

use ic_approx::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn main() -> anyhow::Result<()> {
    let path = bpaf::short('g')
        .long("graph-path")
        .help("File path of a graph edge list.")
        .argument::<PathBuf>("GRAPH_PATH");
    let direction = bpaf::short('d')
        .long("direction")
        .help("Direction of a graph. (directed/undirected)")
        .argument::<Direction>("DIRECTION");
    let separator = bpaf::short('s')
        .long("separator")
        .help("A character to separate columns of an edge list. (default: space)")
        .argument::<u8>("SEPARATOR")
        .fallback(b' ');
    let has_header = bpaf::short('e')
        .long("header")
        .help("Use the first row as a column header.")
        .switch();

    let parser = bpaf::construct!(path, direction, separator, has_header)
        .to_options()
        .version("0.1.0")
        .descr("Experiments of approximate methods for IC model");

    let (path, direction, separator, has_header) = parser.run();
    let GraphData { nnodes, edges, .. } = read_edge_list(path, direction, separator, has_header)?;

    let rng_seed = 0;
    let high_prob = 0.125;
    let nseeds = 10;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng_seed);
    let prb = generate_prob_mat(nnodes, &edges, high_prob, &mut rng);
    let seeds = generate_seeds(nnodes, nseeds, &mut rng);
    let preds_of = preds_of(nnodes, &edges);

    let niter = 5000;
    let adj = adj_binmat(nnodes, &edges);
    let (sim_distr, sim_t, sim_dur) =
        simulate_ic_model(nnodes, &adj, &prb, &seeds, &mut rng, niter);
    println!("Execution time (SIM): {} ms", sim_dur.as_millis());

    let t = sim_t.ceil() as usize;
    println!("{t}");

    let (dmp_distr, dmp_dur) = finite_dmp(nnodes, &edges, &preds_of, &prb, &seeds, t);
    println!("Execution time (DMP): {} ms", dmp_dur.as_millis());

    let (tyr_distr, tyr_dur) =
        finite_taylor_scaled_point(nnodes, &preds_of, prb.transpose(), &seeds, t, 1.0);
    println!("Execution time (TYR): {} ms", tyr_dur.as_millis());

    let (tyr0_distr, tyr0_dur) = finite_taylor_zero_point(nnodes, prb.transpose(), &seeds, t);
    println!("Execution time (TYR0): {} ms", tyr0_dur.as_millis());

    println!(
        "Error (SIM vs DMP ): {:?}",
        error_of_distrs(&sim_distr, &dmp_distr, nnodes)
    );
    println!(
        "Error (SIM vs TYR ): {:?}",
        error_of_distrs(&sim_distr, &tyr_distr, nnodes)
    );
    println!(
        "Error (SIM vs TYR0): {:?}",
        error_of_distrs(&sim_distr, &tyr0_distr, nnodes)
    );
    println!(
        "Error (DMP vs TYR ): {:?}",
        error_of_distrs(&dmp_distr, &tyr_distr, nnodes)
    );
    println!(
        "Error (DMP vs TYR0): {:?}",
        error_of_distrs(&dmp_distr, &tyr0_distr, nnodes)
    );
    println!(
        "Error (TYR vs TYR0): {:?}",
        error_of_distrs(&tyr_distr, &tyr0_distr, nnodes)
    );
    Ok(())
}
