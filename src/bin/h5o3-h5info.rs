//pub mod h5parm;

use clap::Parser;

extern crate h5o3;

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
    /// Verbose output (e.g. the history)
    #[arg(long, default_value("false"))]
    verbose: bool,
}

fn summarise_h5parm(h5parm: &String, solset: String, verbose: bool) {
    let h5name = h5parm.split("/").last().unwrap();
    println!("Summarising {}\n", h5name);
    let h5 = h5o3::H5parm::open(h5parm, false).expect("Failed to read H5parm.");
    println!(
        "{:<26} {:<19} {:<15} {:<11} {:<13}",
        "Solutions", "Type", "Polarisations", "% flagged", "Antennas"
    );
    if solset.len() == 0 {
        for ss in h5.solsets {
            println!("|-{}", ss.name);
            for st in ss.soltabs {
                let stationlist = st.get_antennas();
                let cs = stationlist
                    .iter()
                    .filter(|s| s.starts_with("CS") || s.starts_with("ST"))
                    .collect::<Vec<_>>();
                let rs = stationlist
                    .iter()
                    .filter(|s| s.starts_with("RS"))
                    .collect::<Vec<_>>();
                let is = stationlist
                    .iter()
                    .filter(|s| !s.starts_with("CS") && !s.starts_with("RS"))
                    .collect::<Vec<_>>();
                println!(
                    "|---{:<22} {:<19} {:<15} {:<11.2} {} ({}/{}/{})",
                    st.name,
                    st.get_type(),
                    st.get_polarisations().to_vec().join(","),
                    st.get_flagged_fraction() * 100.0,
                    st.get_antennas().len(),
                    cs.len(),
                    rs.len(),
                    is.len()
                );
                if verbose {
                    let h = st.get_history();
                    if h.len() > 0 {
                        println!("|\t{}", h);
                    }
                    println!("|");
                }
            }
            println!();
        }
    } else {
        let ss = h5.get_solset(solset).unwrap();
        println!("|-{}", ss.name);
        for st in &ss.soltabs {
            let stationlist = st.get_antennas();
            let cs = stationlist
                .iter()
                .filter(|s| s.starts_with("CS"))
                .collect::<Vec<_>>();
            let rs = stationlist
                .iter()
                .filter(|s| s.starts_with("RS"))
                .collect::<Vec<_>>();
            println!(
                "|---{:<22} {:<19} {:<15} {:<11.2} {:<19} ({}/{})",
                st.name,
                st.get_type(),
                st.get_polarisations().to_vec().join(","),
                st.get_flagged_fraction()*100.0,
                st.get_antennas().len(),
                cs.len(),
                rs.len()
            );
            if verbose {
                let h = st.get_history();
                if h.len() > 0 {
                    println!("|\t{}", h);
                }
                println!("|");
            }
        }
    }
}

fn main() {
    let args = Args::parse();
    println!("H5parm: {}\n", args.h5parm);
    summarise_h5parm(&args.h5parm, args.solset, args.verbose);
}
