use super::{DefaultAesthetics, GeomTrait, GeomType, StatResult};
use arrow::datatypes::DataType;
use crate::plot::types::DefaultAestheticValue;
use crate::{naming, Mappings};

#[derive(Debug, Clone, Copy)]
pub struct Spatial;

impl GeomTrait for Spatial {
    fn geom_type(&self) -> GeomType {
        GeomType::Spatial
    }

    fn aesthetics(&self) -> DefaultAesthetics {
        DefaultAesthetics {
            defaults: &[
                ("geometry", DefaultAestheticValue::Required),
                ("fill", DefaultAestheticValue::String("steelblue")),
                ("stroke", DefaultAestheticValue::String("white")),
                ("opacity", DefaultAestheticValue::Number(0.8)),
                ("linewidth", DefaultAestheticValue::Number(0.5)),
                ("linetype", DefaultAestheticValue::String("solid")),
            ],
        }
    }

    fn needs_stat_transform(&self, _aesthetics: &Mappings) -> bool {
        true
    }

    fn apply_stat_transform(
        &self,
        query: &str,
        schema: &crate::plot::Schema,
        _aesthetics: &Mappings,
        _group_by: &[String],
        _parameters: &std::collections::HashMap<String, crate::plot::ParameterValue>,
        _execute_query: &dyn Fn(&str) -> crate::Result<crate::DataFrame>,
        _dialect: &dyn crate::reader::SqlDialect,
    ) -> crate::Result<StatResult> {
        let geom_col = naming::aesthetic_column("geometry");

        // DuckDB GEOMETRY columns arrive as Binary in Arrow (no native geometry type).
        // Convert to WKB via ST_AsWKB so the writer can parse them as standard WKB.
        // String columns (GeoJSON, WKB hex) are handled directly by the writer.
        let is_binary = schema
            .iter()
            .find(|c| c.name == geom_col)
            .map(|c| matches!(c.dtype, DataType::Binary | DataType::LargeBinary))
            .unwrap_or(false);

        if !is_binary {
            return Ok(StatResult::Identity);
        }

        let quoted = naming::quote_ident(&geom_col);
        Ok(StatResult::Transformed {
            query: format!(
                "SELECT * REPLACE (ST_AsWKB({col}) AS {col}) FROM ({query})",
                col = quoted,
                query = query,
            ),
            stat_columns: vec![],
            dummy_columns: vec![],
            consumed_aesthetics: vec![],
        })
    }
}

impl std::fmt::Display for Spatial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spatial")
    }
}
