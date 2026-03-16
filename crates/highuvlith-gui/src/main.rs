mod app;
mod state;

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("highuvlith — VUV Lithography Simulator"),
        ..Default::default()
    };
    eframe::run_native(
        "highuvlith",
        options,
        Box::new(|cc| Ok(Box::new(app::LithApp::new(cc)))),
    )
}
