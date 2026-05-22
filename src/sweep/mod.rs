#![allow(clippy::similar_names, clippy::too_many_lines)]
//! Layer 7 — parameter-grid sweep runner.
//!
//! Consumes `[sweep]` from a config and runs the cartesian product of
//! `axes` as independent solves into per-cell subdirectories.

use std::path::Path;

use crate::bin_support;
use crate::config::Config;
use crate::error::{Result, ScrapboxError};

/// Execute a parameter sweep specified by `[sweep]` in the config at
/// `config_path`. Returns `Ok` if every cell ran to completion.
pub fn run(config_path: &Path) -> Result<()> {
    let raw = std::fs::read_to_string(config_path).map_err(|source| ScrapboxError::ConfigRead {
        path: config_path.to_path_buf(),
        source,
    })?;
    let base_cfg = Config::from_toml_str(&raw)?;
    let sweep = base_cfg
        .sweep
        .as_ref()
        .ok_or_else(|| ScrapboxError::ConfigValidation {
            message: "[sweep] section missing — scrapbox sweep requires it".into(),
        })?;
    if sweep.axes.is_empty() {
        return Err(ScrapboxError::ConfigValidation {
            message: "[sweep].axes must be non-empty".into(),
        });
    }
    if sweep.parallelism == 0 {
        return Err(ScrapboxError::ConfigValidation {
            message: "[sweep].parallelism must be >= 1".into(),
        });
    }

    let base_table: toml::Value =
        raw.parse()
            .map_err(|e: toml::de::Error| ScrapboxError::ConfigValidation {
                message: format!("re-parse of sweep config failed: {e}"),
            })?;

    let cells = cartesian_product(sweep.axes.iter().map(|a| a.values.len()).collect());
    eprintln!(
        "scrapbox sweep: {} cells across {} axes",
        cells.len(),
        sweep.axes.len()
    );

    // Pre-build every cell config sequentially. Collect all build errors
    // rather than short-circuit so a malformed sweep does not hide later
    // failures behind the first one.
    #[allow(clippy::needless_collect)]
    let cell_results: Vec<Result<SweepCell>> = cells
        .iter()
        .enumerate()
        .map(|(idx, cell)| -> Result<SweepCell> {
            let mut patched = base_table.clone();
            for (axis_idx, &value_idx) in cell.iter().enumerate() {
                let axis = &sweep.axes[axis_idx];
                let value = axis.values[value_idx];
                set_dotted_key(&mut patched, &axis.key, toml::Value::Float(value))?;
            }
            let subdir = render_subdir(&sweep.subdir_template, &sweep.axes, cell)?;
            set_dotted_key(
                &mut patched,
                "output.directory",
                toml::Value::String(subdir.clone()),
            )?;
            let patched_str =
                toml::to_string(&patched).map_err(|e| ScrapboxError::ConfigValidation {
                    message: format!("re-serialize of patched sweep cell failed: {e}"),
                })?;
            let cell_cfg = Config::from_toml_str(&patched_str)?;
            Ok(SweepCell {
                idx,
                summary: cell_summary(&sweep.axes, cell),
                subdir,
                config: cell_cfg,
            })
        })
        .collect();
    let (oks, errs): (Vec<_>, Vec<_>) = cell_results.into_iter().partition(Result::is_ok);
    if !errs.is_empty() {
        for e in &errs {
            if let Err(err) = e {
                eprintln!("  sweep cell config failed: {err}");
            }
        }
        return Err(errs
            .into_iter()
            .next()
            .and_then(Result::err)
            .expect("partition guarantees at least one Err"));
    }
    let cell_configs: Vec<SweepCell> = oks.into_iter().map(Result::unwrap).collect();

    // Pre-flight: subdir collisions would silently lose data via
    // overwrite=true. Fail fast if two cells render to the same path.
    let mut seen = std::collections::HashSet::new();
    for cell in &cell_configs {
        if !seen.insert(cell.subdir.as_str()) {
            return Err(ScrapboxError::ConfigValidation {
                message: format!(
                    "sweep subdir collision: {} appears in multiple cells",
                    cell.subdir
                ),
            });
        }
    }

    let total = cell_configs.len();
    // Both branches share semantics: every cell writes to a distinct
    // subdir (verified above), so concurrent I/O is safe. A dedicated
    // rayon ThreadPoolBuilder honours the user's parallelism cap and
    // avoids contaminating the global pool that faer uses for dense diag.
    let results: Vec<Result<()>> = if sweep.parallelism == 1 {
        cell_configs
            .iter()
            .map(|cell| {
                eprintln!(
                    "  cell {}/{}: {} -> {}",
                    cell.idx + 1,
                    total,
                    cell.summary,
                    cell.subdir
                );
                bin_support::solve_and_write(&cell.config)
            })
            .collect()
    } else {
        use rayon::prelude::*;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(sweep.parallelism)
            .build()
            .map_err(|e| ScrapboxError::ConfigValidation {
                message: format!("rayon thread pool build failed: {e}"),
            })?;
        pool.install(|| {
            cell_configs
                .par_iter()
                .map(|cell| {
                    eprintln!(
                        "  cell {}/{}: {} -> {} (parallel)",
                        cell.idx + 1,
                        total,
                        cell.summary,
                        cell.subdir
                    );
                    bin_support::solve_and_write(&cell.config)
                })
                .collect()
        })
    };
    // Collect ALL errors rather than short-circuit - rayon already ran
    // every cell to completion, so dropping later errors would force
    // users to fix-rerun-discover for each failing cell.
    let errors: Vec<ScrapboxError> = results.into_iter().filter_map(Result::err).collect();
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("  cell failure: {e}");
        }
        return Err(errors.into_iter().next().expect("checked non-empty"));
    }
    Ok(())
}

#[derive(Debug)]
struct SweepCell {
    idx: usize,
    summary: String,
    subdir: String,
    config: Config,
}

fn cartesian_product(lens: Vec<usize>) -> Vec<Vec<usize>> {
    let mut out = vec![vec![]];
    for n in lens {
        let mut next = Vec::with_capacity(out.len() * n);
        for prefix in &out {
            for i in 0..n {
                let mut e = prefix.clone();
                e.push(i);
                next.push(e);
            }
        }
        out = next;
    }
    out
}

fn set_dotted_key(root: &mut toml::Value, key: &str, value: toml::Value) -> Result<()> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        return Err(ScrapboxError::ConfigValidation {
            message: "sweep axis key is empty".into(),
        });
    }
    let mut cursor = root;
    for part in &parts[..parts.len() - 1] {
        let table = cursor
            .as_table_mut()
            .ok_or_else(|| ScrapboxError::ConfigValidation {
                message: format!("sweep key {key} traverses a non-table at {part}"),
            })?;
        cursor = table
            .entry((*part).to_string())
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    }
    let table = cursor
        .as_table_mut()
        .ok_or_else(|| ScrapboxError::ConfigValidation {
            message: format!("sweep key {key} final cursor is not a table"),
        })?;
    table.insert(parts[parts.len() - 1].to_string(), value);
    Ok(())
}

fn render_subdir(
    template: &str,
    axes: &[crate::config::SweepAxis],
    cell: &[usize],
) -> Result<String> {
    let mut out = template.to_string();
    for (axis_idx, &value_idx) in cell.iter().enumerate() {
        let axis = &axes[axis_idx];
        let value = axis.values[value_idx];
        let placeholder = format!("{{{label}}}", label = axis.effective_label());
        let replacement = format_axis_value(value);
        if !out.contains(&placeholder) {
            return Err(ScrapboxError::ConfigValidation {
                message: format!(
                    "subdir_template {template} does not reference axis label {label}",
                    label = axis.effective_label()
                ),
            });
        }
        out = out.replace(&placeholder, &replacement);
    }
    Ok(out)
}

fn format_axis_value(v: f64) -> String {
    let raw = format!("{v:.6}");
    let trimmed = raw.trim_end_matches('0').trim_end_matches('.');
    trimmed.replace('-', "m").replace('.', "p")
}

fn cell_summary(axes: &[crate::config::SweepAxis], cell: &[usize]) -> String {
    cell.iter()
        .enumerate()
        .map(|(i, &v_idx)| format!("{}={}", axes[i].effective_label(), axes[i].values[v_idx]))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartesian_product_zero_axes() {
        let out = cartesian_product(vec![]);
        assert_eq!(out, vec![Vec::<usize>::new()]);
    }

    #[test]
    fn cartesian_product_two_by_three_yields_six_cells() {
        let out = cartesian_product(vec![2, 3]);
        assert_eq!(out.len(), 6);
        assert_eq!(out[0], vec![0, 0]);
        assert_eq!(out[5], vec![1, 2]);
    }

    #[test]
    fn set_dotted_key_writes_into_nested_table() {
        let mut root: toml::Value = "[a.b]\nc = 1\n".parse().unwrap();
        set_dotted_key(&mut root, "a.b.c", toml::Value::Integer(99)).unwrap();
        assert_eq!(root["a"]["b"]["c"].as_integer(), Some(99));
    }

    #[test]
    fn set_dotted_key_creates_missing_tables() {
        let mut root: toml::Value = "".parse().unwrap();
        set_dotted_key(&mut root, "x.y.z", toml::Value::Float(2.5)).unwrap();
        assert_eq!(root["x"]["y"]["z"].as_float(), Some(2.5));
    }

    #[test]
    fn format_axis_value_filesystem_safe() {
        assert_eq!(format_axis_value(2.5), "2p5");
        assert_eq!(format_axis_value(-0.1), "m0p1");
        assert_eq!(format_axis_value(4.0), "4");
    }

    #[test]
    fn render_subdir_substitutes_labels() {
        let axes = vec![
            crate::config::SweepAxis {
                key: "hamiltonian.on_site_interaction".into(),
                label: Some("U".into()),
                values: vec![2.0, 4.0, 6.0],
            },
            crate::config::SweepAxis {
                key: "hamiltonian.beta".into(),
                label: None,
                values: vec![1.0, 2.0],
            },
        ];
        let s = render_subdir("U_{U}_beta_{beta}", &axes, &[1, 0]).unwrap();
        assert_eq!(s, "U_4_beta_1");
    }

    #[test]
    fn render_subdir_errors_on_missing_label() {
        let axes = vec![crate::config::SweepAxis {
            key: "hamiltonian.beta".into(),
            label: None,
            values: vec![1.0],
        }];
        let err = render_subdir("nope", &axes, &[0]).unwrap_err();
        assert!(format!("{err}").contains("subdir_template"));
    }
}
