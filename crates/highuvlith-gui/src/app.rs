use crate::state::{SimResult, SimState};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

pub struct LithApp {
    state: SimState,
    show_cross_section: bool,
}

impl LithApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self {
            state: SimState::new(),
            show_cross_section: true,
        };
        app.state.try_compute();
        app
    }
}

impl eframe::App for LithApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.state.try_compute();

        // Request repaint while computing
        if *self.state.computing.lock().unwrap() {
            ctx.request_repaint();
        }

        // Left panel: parameters
        egui::SidePanel::left("params_panel")
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.heading("VUV Lithography Simulator");
                ui.separator();
                self.draw_params(ui);
            });

        // Bottom panel: status
        egui::TopBottomPanel::bottom("status_panel").show(ctx, |ui| {
            self.draw_status(ui);
        });

        // Central panel: visualization
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_visualization(ui);
        });
    }
}

impl LithApp {
    fn draw_params(&mut self, ui: &mut egui::Ui) {
        let p = &mut self.state.params;
        let mut changed = false;

        egui::CollapsingHeader::new("Source")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("\u{03bb} (nm):");
                    changed |= ui
                        .add(egui::Slider::new(&mut p.wavelength_nm, 120.0..=170.0).step_by(0.1))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("\u{03c3}:");
                    changed |= ui
                        .add(egui::Slider::new(&mut p.sigma, 0.1..=1.0).step_by(0.05))
                        .changed();
                });
            });

        egui::CollapsingHeader::new("Optics")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("NA:");
                    changed |= ui
                        .add(egui::Slider::new(&mut p.na, 0.3..=0.95).step_by(0.01))
                        .changed();
                });
            });

        egui::CollapsingHeader::new("Mask")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("CD (nm):");
                    changed |= ui
                        .add(egui::Slider::new(&mut p.cd_nm, 20.0..=200.0).step_by(1.0))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Pitch (nm):");
                    changed |= ui
                        .add(egui::Slider::new(&mut p.pitch_nm, 40.0..=500.0).step_by(5.0))
                        .changed();
                });
            });

        egui::CollapsingHeader::new("Process")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Focus (nm):");
                    changed |= ui
                        .add(egui::Slider::new(&mut p.focus_nm, -500.0..=500.0).step_by(5.0))
                        .changed();
                });
            });

        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Grid:");
            let sizes = [64usize, 128, 256];
            for &s in &sizes {
                if ui
                    .selectable_label(p.grid_size == s, format!("{}", s))
                    .clicked()
                {
                    p.grid_size = s;
                    changed = true;
                }
            }
        });

        if changed {
            self.state.mark_dirty();
        }
    }

    fn draw_status(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let is_computing = *self.state.computing.lock().unwrap();
            if is_computing {
                ui.spinner();
                ui.label("Computing...");
            } else if let Some(ref result) = *self.state.result.lock().unwrap() {
                ui.label(format!(
                    "Grid: {}x{} | Kernels: {} | {:.1}ms | Contrast: {:.4} | I: [{:.4}, {:.4}]",
                    self.state.params.grid_size,
                    self.state.params.grid_size,
                    result.num_kernels,
                    result.compute_ms,
                    result.contrast,
                    result.i_min,
                    result.i_max,
                ));
            }
        });
    }

    fn draw_visualization(&mut self, ui: &mut egui::Ui) {
        let has_result = self.state.result.lock().unwrap().is_some();
        if !has_result {
            ui.centered_and_justified(|ui| {
                ui.label("Initializing simulation...");
            });
            return;
        }

        // Tab selector (no lock held, so &mut self is fine)
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.show_cross_section, false, "Aerial Image");
            ui.selectable_value(&mut self.show_cross_section, true, "Cross-Section");
        });
        ui.separator();

        let result_lock = self.state.result.lock().unwrap();
        let result = result_lock.as_ref().unwrap();

        if self.show_cross_section {
            self.draw_cross_section_inner(ui, result);
        } else {
            self.draw_aerial_image_inner(ui, result);
        }
    }

    fn draw_aerial_image_inner(&self, ui: &mut egui::Ui, result: &SimResult) {
        let n = result.intensity.ncols();
        let size = [n, n];

        // Convert intensity to grayscale pixels (inferno-like colormap)
        let mut pixels = vec![egui::Color32::BLACK; n * n];
        let range = result.i_max - result.i_min;
        if range > 1e-15 {
            for i in 0..n {
                for j in 0..n {
                    let val = (result.intensity[[i, j]] - result.i_min) / range;
                    let val = val.clamp(0.0, 1.0);
                    pixels[i * n + j] = inferno_color(val);
                }
            }
        }

        let image = egui::ColorImage { size, pixels };
        let texture = ui
            .ctx()
            .load_texture("aerial", image, egui::TextureOptions::NEAREST);

        let available = ui.available_size();
        let img_size = available.min_elem().min(600.0);
        ui.centered_and_justified(|ui| {
            ui.add(
                egui::Image::from_texture(&texture)
                    .fit_to_exact_size(egui::Vec2::splat(img_size)),
            );
        });
    }

    fn draw_cross_section_inner(&self, ui: &mut egui::Ui, result: &SimResult) {
        let points: PlotPoints = result
            .cross_section_x
            .iter()
            .zip(result.cross_section_i.iter())
            .map(|(&x, &i)| [x, i])
            .collect();

        let line = Line::new(points)
            .color(egui::Color32::from_rgb(50, 120, 255))
            .width(2.0);

        Plot::new("cross_section")
            .x_axis_label("x (nm)")
            .y_axis_label("Intensity")
            .show(ui, |plot_ui| {
                plot_ui.line(line);
            });
    }
}

/// Simple inferno-like colormap: black -> purple -> orange -> yellow
fn inferno_color(t: f64) -> egui::Color32 {
    let t = t as f32;
    let r;
    let g;
    let b;

    if t < 0.25 {
        let s = t / 0.25;
        r = (s * 80.0) as u8;
        g = 0;
        b = (s * 120.0) as u8;
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        r = (80.0 + s * 140.0) as u8;
        g = (s * 30.0) as u8;
        b = (120.0 - s * 70.0) as u8;
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        r = (220.0 + s * 35.0) as u8;
        g = (30.0 + s * 130.0) as u8;
        b = (50.0 - s * 50.0) as u8;
    } else {
        let s = (t - 0.75) / 0.25;
        r = 255;
        g = (160.0 + s * 95.0) as u8;
        b = (s * 100.0) as u8;
    }

    egui::Color32::from_rgb(r, g, b)
}
