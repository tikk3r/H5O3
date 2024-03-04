//pub mod h5parm;

use clap::Parser;

extern crate lofar_h5parm_rs;

/// A Rust interface to summarise LOFAR H5parm calibration tables.
#[derive(Parser, Debug)]
#[command(name = "LOFAR-H5info")]
#[command(author = "Frits Sweijen")]
#[command(version = "0.0.0")]
#[command(
    help_template = "{name} \nVersion: {version} \nAuthor: {author}\n{about-section} \n {usage-heading} {usage} \n {all-args} {tab}"
)]
// #[clap(author="Author Name", version, about="")]
struct Args {
    /// H5parm to summarise.
    #[arg(long)]
    h5parm: String,
    /// SolSet to display.
    #[arg(long, default_value = "")]
    solset: String,
}

fn summarise_h5parm(h5parm: &String) {
    let h5name = h5parm.split("/").last().unwrap();
    println!("Summarising {}\n", h5name);
    let h5 = lofar_h5parm_rs::H5parm::open(h5parm, false).expect("Failed to read H5parm.");
    println!(
        "{:<26} {:<19} {:<15} {:<11} {:<19}",
        "Solutions", "Type", "Polarisations", "% flagged", "Antennas"
    );
    for ss in h5.solsets {
        println!("|-{}", ss.name);
        for st in ss.soltabs {
            if st.is_fulljones {
                println!(
                    "|---{:<15} Full-Jones: {} {}",
                    st.name,
                    st.is_fulljones,
                    st.get_polarisations()
                );
            } else {
                println!(
                    "|---{:<22} {:<19} {:<15} {:<11} {:<19}",
                    st.name,
                    st.get_type(),
                    st.get_polarisations().to_vec().join(","),
                    st.get_flagged_fraction()
                        .iter()
                        .map(|m| format!("{:.3}%", m * 100.0))
                        .collect::<Vec<_>>()
                        .join(","),
                    st.get_antennas().len()
                );
            }
            let h = st.get_history();
            if h.len() > 0 {
                println!("|\t{}", h);
            }
            println!("|");
        }
        println!();
    }
}

fn main() {
    let args = Args::parse();
    println!("H5parm: {}\n", args.h5parm);
    summarise_h5parm(&args.h5parm);
}
