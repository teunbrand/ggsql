//! Smooth geom implementation

use super::{DefaultAesthetics, GeomTrait, GeomType};
use crate::plot::types::DefaultAestheticValue;
use crate::plot::DefaultParam;
use crate::Mappings;

/// Smooth geom - smoothed conditional means (regression, LOESS, etc.)
#[derive(Debug, Clone, Copy)]
pub struct Smooth;

impl GeomTrait for Smooth {
    fn geom_type(&self) -> GeomType {
        GeomType::Smooth
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("pos1", DefaultAestheticValue::Required),
                ("pos2", DefaultAestheticValue::Required),
                ("weight", DefaultAestheticValue::Null),
                ("stroke", DefaultAestheticValue::String("#3366FF")),
                ("linewidth", DefaultAestheticValue::Number(2.0)),
                ("opacity", DefaultAestheticValue::Number(1.0)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn default_params(&self) -> &'static [super::DefaultParam] {
        &[
            DefaultParam {
                name: "method",
                default: super::DefaultParamValue::String("nw"),
            },
            DefaultParam {
                name: "bandwidth",
                default: super::DefaultParamValue::Null,
            },
            DefaultParam {
                name: "adjust",
                default: super::DefaultParamValue::Number(1.0),
            },
            DefaultParam {
                name: "kernel",
                default: super::DefaultParamValue::String("gaussian"),
            },
        ]
    }

    fn default_remappings(&self) -> &'static [(&'static str, DefaultAestheticValue)] {
        &[
            ("pos1", DefaultAestheticValue::Column("pos1")),
            ("pos2", DefaultAestheticValue::Column("intensity")),
        ]
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        _schema: &crate::plot::Schema,
        aesthetics: &Mappings,
        group_by: &[String],
        parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
        execute_query: &dyn Fn(&str) -> crate::Result<polars::prelude::DataFrame>,
    ) -> crate::Result<super::StatResult> {
        super::density::stat_density(
            query,
            aesthetics,
            "pos1",
            Some("pos2"),
            group_by,
            parameters,
            execute_query,
        )
    }

    // Note: stat_smooth not yet implemented - will return Identity for now
}

impl std::fmt::Display for Smooth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "smooth")
    }
}
