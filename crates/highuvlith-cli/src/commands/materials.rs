use highuvlith_core::materials::database::MaterialsDatabase;

pub fn run(wavelength: Option<f64>, name: Option<&str>) -> anyhow::Result<()> {
    let db = MaterialsDatabase::new();
    let wl = wavelength.unwrap_or(157.0);

    let materials = [
        "CaF2", "MgF2", "LiF", "BaF2", "SiO2",
        "Cr", "Si", "AlF3", "Na3AlF6", "LaF3", "GdF3",
        "VUV_resist", "VUV_BARC",
    ];

    if let Some(mat_name) = name {
        match db.refractive_index(mat_name, wl) {
            Ok(n) => {
                println!("{} at {:.1} nm:", mat_name, wl);
                println!("  n = {:.4}", n.re);
                println!("  k = {:.4}", n.im);
                if let Ok(disp) = db.dispersion(mat_name, wl) {
                    println!("  dn/dλ = {:.6} /nm", disp);
                }
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    } else {
        println!("{:<12} {:>8} {:>8}  (at {:.1} nm)", "Material", "n", "k", wl);
        println!("{}", "-".repeat(40));
        for mat in &materials {
            match db.refractive_index(mat, wl) {
                Ok(n) => println!("{:<12} {:>8.4} {:>8.4}", mat, n.re, n.im),
                Err(_) => println!("{:<12} {:>8} {:>8}", mat, "N/A", "N/A"),
            }
        }
    }

    Ok(())
}
