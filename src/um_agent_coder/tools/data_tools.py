"""
Specialized tools for data engineering, data science, and ML tasks.
These tools handle SQL operations, data profiling, schema analysis, and pipeline building.
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import re
import yaml
from datetime import datetime

from .base import Tool, ToolResult


@dataclass
class SchemaInfo:
    """Information about a database schema."""
    table_name: str
    columns: List[Dict[str, Any]]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    indexes: List[str]
    row_count: Optional[int] = None
    size_mb: Optional[float] = None


@dataclass
class DataProfile:
    """Statistical profile of a dataset."""
    column_name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    min_value: Any
    max_value: Any
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_dev: Optional[float] = None
    percentiles: Optional[Dict[str, float]] = None
    top_values: Optional[List[Tuple[Any, int]]] = None


class SQLExecutor(Tool):
    """Execute SQL queries and DDL statements."""
    
    def __init__(self):
        super().__init__(
            name="SQLExecutor",
            description="Execute SQL queries with support for multiple databases"
        )
    
    def execute(
        self,
        query: str,
        connection_string: Optional[str] = None,
        database_type: str = "sqlite",
        return_df: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute SQL query."""
        try:
            if database_type == "sqlite":
                import sqlite3
                conn = sqlite3.connect(connection_string or ":memory:")
            elif database_type == "postgresql":
                import psycopg2
                conn = psycopg2.connect(connection_string)
            elif database_type == "mysql":
                import mysql.connector
                conn = mysql.connector.connect(connection_string)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported database type: {database_type}"
                )
            
            if return_df and query.strip().upper().startswith("SELECT"):
                df = pd.read_sql_query(query, conn)
                result_data = {
                    "dataframe": df.to_dict('records'),
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict()
                }
            else:
                cursor = conn.cursor()
                cursor.execute(query)
                conn.commit()
                result_data = {
                    "rows_affected": cursor.rowcount,
                    "success": True
                }
            
            conn.close()
            
            return ToolResult(
                success=True,
                data=result_data
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"SQL execution error: {str(e)}"
            )


class SchemaAnalyzer(Tool):
    """Analyze database schemas and relationships."""
    
    def __init__(self):
        super().__init__(
            name="SchemaAnalyzer",
            description="Analyze database schemas, relationships, and constraints"
        )
    
    def execute(
        self,
        connection_string: Optional[str] = None,
        database_type: str = "sqlite",
        schema_name: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Analyze database schema."""
        try:
            schemas = []
            
            if database_type == "sqlite":
                import sqlite3
                conn = sqlite3.connect(connection_string or ":memory:")
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    
                    # Get columns
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    column_info = []
                    primary_keys = []
                    
                    for col in columns:
                        col_dict = {
                            "name": col[1],
                            "type": col[2],
                            "nullable": not col[3],
                            "default": col[4],
                            "primary_key": bool(col[5])
                        }
                        column_info.append(col_dict)
                        if col[5]:
                            primary_keys.append(col[1])
                    
                    # Get foreign keys
                    cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                    fks = cursor.fetchall()
                    foreign_keys = [
                        {
                            "column": fk[3],
                            "referenced_table": fk[2],
                            "referenced_column": fk[4]
                        }
                        for fk in fks
                    ]
                    
                    # Get indexes
                    cursor.execute(f"PRAGMA index_list({table_name})")
                    indexes = [idx[1] for idx in cursor.fetchall()]
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    
                    schemas.append(SchemaInfo(
                        table_name=table_name,
                        columns=column_info,
                        primary_keys=primary_keys,
                        foreign_keys=foreign_keys,
                        indexes=indexes,
                        row_count=row_count
                    ))
                
                conn.close()
            
            # Generate relationships diagram
            relationships = self._generate_relationships(schemas)
            
            return ToolResult(
                success=True,
                data={
                    "schemas": [vars(s) for s in schemas],
                    "relationships": relationships,
                    "table_count": len(schemas),
                    "total_columns": sum(len(s.columns) for s in schemas)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Schema analysis error: {str(e)}"
            )
    
    def _generate_relationships(self, schemas: List[SchemaInfo]) -> List[Dict[str, str]]:
        """Generate relationship map between tables."""
        relationships = []
        
        for schema in schemas:
            for fk in schema.foreign_keys:
                relationships.append({
                    "from_table": schema.table_name,
                    "from_column": fk["column"],
                    "to_table": fk["referenced_table"],
                    "to_column": fk["referenced_column"],
                    "type": "foreign_key"
                })
        
        return relationships


class DataProfiler(Tool):
    """Profile data to understand distributions and quality."""
    
    def __init__(self):
        super().__init__(
            name="DataProfiler",
            description="Generate statistical profiles and quality reports for datasets"
        )
    
    def execute(
        self,
        data_source: Union[str, pd.DataFrame],
        sample_size: Optional[int] = None,
        profile_all: bool = True,
        **kwargs
    ) -> ToolResult:
        """Profile dataset."""
        try:
            # Load data
            if isinstance(data_source, str):
                if data_source.endswith('.csv'):
                    df = pd.read_csv(data_source, nrows=sample_size)
                elif data_source.endswith('.parquet'):
                    df = pd.read_parquet(data_source)
                    if sample_size:
                        df = df.sample(min(sample_size, len(df)))
                elif data_source.endswith('.json'):
                    df = pd.read_json(data_source)
                    if sample_size:
                        df = df.sample(min(sample_size, len(df)))
                else:
                    return ToolResult(
                        success=False,
                        error=f"Unsupported file format: {data_source}"
                    )
            else:
                df = data_source
            
            profiles = []
            
            for column in df.columns:
                col_data = df[column]
                
                profile = DataProfile(
                    column_name=column,
                    data_type=str(col_data.dtype),
                    null_count=col_data.isna().sum(),
                    null_percentage=col_data.isna().sum() / len(df) * 100,
                    unique_count=col_data.nunique(),
                    unique_percentage=col_data.nunique() / len(df) * 100,
                    min_value=None,
                    max_value=None
                )
                
                # Numeric columns
                if pd.api.types.is_numeric_dtype(col_data):
                    profile.min_value = col_data.min()
                    profile.max_value = col_data.max()
                    profile.mean_value = col_data.mean()
                    profile.median_value = col_data.median()
                    profile.std_dev = col_data.std()
                    profile.percentiles = {
                        "25%": col_data.quantile(0.25),
                        "50%": col_data.quantile(0.50),
                        "75%": col_data.quantile(0.75),
                        "95%": col_data.quantile(0.95),
                        "99%": col_data.quantile(0.99)
                    }
                
                # Categorical columns
                elif pd.api.types.is_object_dtype(col_data):
                    value_counts = col_data.value_counts()
                    profile.top_values = list(value_counts.head(10).items())
                    profile.min_value = col_data.min() if not col_data.empty else None
                    profile.max_value = col_data.max() if not col_data.empty else None
                
                # Datetime columns
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    profile.min_value = col_data.min()
                    profile.max_value = col_data.max()
                
                profiles.append(profile)
            
            # Data quality issues
            quality_issues = self._detect_quality_issues(df, profiles)
            
            return ToolResult(
                success=True,
                data={
                    "profiles": [vars(p) for p in profiles],
                    "dataset_info": {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                        "duplicates": df.duplicated().sum()
                    },
                    "quality_issues": quality_issues
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Data profiling error: {str(e)}"
            )
    
    def _detect_quality_issues(self, df: pd.DataFrame, profiles: List[DataProfile]) -> List[Dict[str, Any]]:
        """Detect common data quality issues."""
        issues = []
        
        for profile in profiles:
            # High null percentage
            if profile.null_percentage > 50:
                issues.append({
                    "column": profile.column_name,
                    "issue": "high_nulls",
                    "severity": "high",
                    "details": f"{profile.null_percentage:.1f}% null values"
                })
            
            # Low cardinality
            if profile.unique_percentage < 1 and len(df) > 100:
                issues.append({
                    "column": profile.column_name,
                    "issue": "low_cardinality",
                    "severity": "medium",
                    "details": f"Only {profile.unique_count} unique values"
                })
            
            # Potential PII
            if any(keyword in profile.column_name.lower() 
                   for keyword in ['ssn', 'email', 'phone', 'address', 'name']):
                issues.append({
                    "column": profile.column_name,
                    "issue": "potential_pii",
                    "severity": "high",
                    "details": "Column may contain personally identifiable information"
                })
        
        return issues


class PipelineBuilder(Tool):
    """Build data pipeline configurations for various orchestrators."""
    
    def __init__(self):
        super().__init__(
            name="PipelineBuilder",
            description="Generate data pipeline configurations for Airflow, dbt, etc."
        )
    
    def execute(
        self,
        pipeline_type: str,
        pipeline_name: str,
        tasks: List[Dict[str, Any]],
        schedule: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Build pipeline configuration."""
        try:
            if pipeline_type == "airflow":
                config = self._build_airflow_dag(pipeline_name, tasks, schedule)
            elif pipeline_type == "dbt":
                config = self._build_dbt_project(pipeline_name, tasks)
            elif pipeline_type == "prefect":
                config = self._build_prefect_flow(pipeline_name, tasks, schedule)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported pipeline type: {pipeline_type}"
                )
            
            # Save configuration
            output_path = Path(f"{pipeline_name}_{pipeline_type}_config")
            output_path.mkdir(exist_ok=True)
            
            for filename, content in config.items():
                file_path = output_path / filename
                file_path.write_text(content)
            
            return ToolResult(
                success=True,
                data={
                    "pipeline_type": pipeline_type,
                    "pipeline_name": pipeline_name,
                    "files_created": list(config.keys()),
                    "output_path": str(output_path)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Pipeline building error: {str(e)}"
            )
    
    def _build_airflow_dag(self, name: str, tasks: List[Dict], schedule: Optional[str]) -> Dict[str, str]:
        """Generate Airflow DAG configuration."""
        dag_code = f'''"""
{name} - Generated Airflow DAG
Generated on {datetime.now().isoformat()}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.sql import SQLOperator
from airflow.operators.bash import BashOperator

default_args = {{
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}}

dag = DAG(
    '{name}',
    default_args=default_args,
    description='Generated data pipeline',
    schedule_interval='{schedule or "@daily"}',
    catchup=False,
    tags=['generated', 'data_pipeline'],
)

'''
        
        # Add tasks
        for i, task in enumerate(tasks):
            task_name = task.get('name', f'task_{i}')
            task_type = task.get('type', 'python')
            
            if task_type == 'sql':
                dag_code += f'''
{task_name} = SQLOperator(
    task_id='{task_name}',
    sql="""
    {task.get('sql', 'SELECT 1')}
    """,
    dag=dag,
)
'''
            elif task_type == 'bash':
                dag_code += f'''
{task_name} = BashOperator(
    task_id='{task_name}',
    bash_command='{task.get('command', 'echo "Task executed"')}',
    dag=dag,
)
'''
            else:
                dag_code += f'''
def {task_name}_func(**context):
    """Execute {task_name}"""
    # Add your logic here
    pass

{task_name} = PythonOperator(
    task_id='{task_name}',
    python_callable={task_name}_func,
    dag=dag,
)
'''
        
        # Add dependencies
        if len(tasks) > 1:
            dag_code += "\n# Task dependencies\n"
            for i in range(len(tasks) - 1):
                task_name = tasks[i].get('name', f'task_{i}')
                next_task = tasks[i + 1].get('name', f'task_{i + 1}')
                dag_code += f"{task_name} >> {next_task}\n"
        
        return {f"{name}_dag.py": dag_code}
    
    def _build_dbt_project(self, name: str, tasks: List[Dict]) -> Dict[str, str]:
        """Generate dbt project structure."""
        dbt_project = f"""
name: '{name}'
version: '1.0.0'
config-version: 2

profile: '{name}'

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["data"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

models:
  {name}:
    staging:
      +materialized: view
    marts:
      +materialized: table
"""
        
        models = {}
        for task in tasks:
            if task.get('type') == 'transform':
                model_name = task.get('name', 'model')
                sql = task.get('sql', f'SELECT * FROM source_table')
                
                models[f"models/{model_name}.sql"] = f"""
-- {model_name} model
-- Generated on {datetime.now().isoformat()}

{{{{ config(
    materialized='table'
) }}}}

{sql}
"""
        
        return {
            "dbt_project.yml": dbt_project,
            **models
        }
    
    def _build_prefect_flow(self, name: str, tasks: List[Dict], schedule: Optional[str]) -> Dict[str, str]:
        """Generate Prefect flow configuration."""
        flow_code = f'''"""
{name} - Generated Prefect Flow
Generated on {datetime.now().isoformat()}
"""

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def extract_data():
    """Extract data from source"""
    # Add extraction logic
    return pd.DataFrame()

@task
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform data"""
    # Add transformation logic
    return df

@task
def load_data(df: pd.DataFrame):
    """Load data to destination"""
    # Add loading logic
    pass

@flow(name="{name}")
def {name}_flow():
    """Main pipeline flow"""
    
'''
        
        # Add tasks to flow
        for i, task in enumerate(tasks):
            task_name = task.get('name', f'task_{i}')
            flow_code += f'    # {task_name}\n'
            flow_code += f'    # Add implementation for {task_name}\n\n'
        
        flow_code += '''
    # Execute pipeline
    raw_data = extract_data()
    transformed_data = transform_data(raw_data)
    load_data(transformed_data)

if __name__ == "__main__":
    {name}_flow()
'''
        
        return {f"{name}_flow.py": flow_code}


class DataValidator(Tool):
    """Validate data quality and business rules."""
    
    def __init__(self):
        super().__init__(
            name="DataValidator",
            description="Validate data against quality rules and constraints"
        )
    
    def execute(
        self,
        data_source: Union[str, pd.DataFrame],
        rules: List[Dict[str, Any]],
        **kwargs
    ) -> ToolResult:
        """Validate data against rules."""
        try:
            # Load data
            if isinstance(data_source, str):
                if data_source.endswith('.csv'):
                    df = pd.read_csv(data_source)
                elif data_source.endswith('.parquet'):
                    df = pd.read_parquet(data_source)
                else:
                    return ToolResult(
                        success=False,
                        error=f"Unsupported file format: {data_source}"
                    )
            else:
                df = data_source
            
            validation_results = []
            
            for rule in rules:
                rule_type = rule.get('type')
                column = rule.get('column')
                
                if rule_type == 'not_null':
                    nulls = df[column].isna().sum()
                    passed = nulls == 0
                    validation_results.append({
                        "rule": f"{column} not null",
                        "passed": passed,
                        "details": f"{nulls} null values found" if not passed else "No nulls"
                    })
                
                elif rule_type == 'unique':
                    duplicates = df[column].duplicated().sum()
                    passed = duplicates == 0
                    validation_results.append({
                        "rule": f"{column} unique",
                        "passed": passed,
                        "details": f"{duplicates} duplicates found" if not passed else "All unique"
                    })
                
                elif rule_type == 'range':
                    min_val = rule.get('min')
                    max_val = rule.get('max')
                    out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
                    passed = out_of_range == 0
                    validation_results.append({
                        "rule": f"{column} in range [{min_val}, {max_val}]",
                        "passed": passed,
                        "details": f"{out_of_range} values out of range" if not passed else "All in range"
                    })
                
                elif rule_type == 'regex':
                    pattern = rule.get('pattern')
                    non_matching = (~df[column].astype(str).str.match(pattern)).sum()
                    passed = non_matching == 0
                    validation_results.append({
                        "rule": f"{column} matches pattern",
                        "passed": passed,
                        "details": f"{non_matching} non-matching values" if not passed else "All match"
                    })
                
                elif rule_type == 'custom':
                    expression = rule.get('expression')
                    failed = (~df.eval(expression)).sum()
                    passed = failed == 0
                    validation_results.append({
                        "rule": expression,
                        "passed": passed,
                        "details": f"{failed} rows failed" if not passed else "All passed"
                    })
            
            # Summary
            total_rules = len(validation_results)
            passed_rules = sum(1 for r in validation_results if r['passed'])
            
            return ToolResult(
                success=True,
                data={
                    "validation_results": validation_results,
                    "summary": {
                        "total_rules": total_rules,
                        "passed": passed_rules,
                        "failed": total_rules - passed_rules,
                        "pass_rate": passed_rules / total_rules * 100 if total_rules > 0 else 0
                    },
                    "dataset_info": {
                        "rows": len(df),
                        "columns": len(df.columns)
                    }
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Data validation error: {str(e)}"
            )


class DimensionalModeler(Tool):
    """Design dimensional models (star/snowflake schemas)."""
    
    def __init__(self):
        super().__init__(
            name="DimensionalModeler",
            description="Design dimensional models for data warehouses"
        )
    
    def execute(
        self,
        business_process: str,
        grain: str,
        dimensions: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        model_type: str = "star",
        **kwargs
    ) -> ToolResult:
        """Design dimensional model."""
        try:
            model = {
                "business_process": business_process,
                "grain": grain,
                "model_type": model_type,
                "fact_table": self._design_fact_table(business_process, facts, dimensions),
                "dimension_tables": [self._design_dimension(dim) for dim in dimensions],
                "relationships": self._define_relationships(business_process, dimensions)
            }
            
            # Generate DDL
            ddl_statements = self._generate_ddl(model)
            
            # Generate documentation
            documentation = self._generate_documentation(model)
            
            return ToolResult(
                success=True,
                data={
                    "model": model,
                    "ddl": ddl_statements,
                    "documentation": documentation,
                    "metrics": {
                        "dimension_count": len(dimensions),
                        "fact_count": len(facts),
                        "total_attributes": sum(len(d.get('attributes', [])) for d in dimensions)
                    }
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Dimensional modeling error: {str(e)}"
            )
    
    def _design_fact_table(self, process: str, facts: List[Dict], dimensions: List[Dict]) -> Dict:
        """Design the fact table."""
        fact_table = {
            "name": f"fact_{process.lower().replace(' ', '_')}",
            "columns": []
        }
        
        # Add surrogate keys for dimensions
        for dim in dimensions:
            dim_name = dim['name'].lower().replace(' ', '_')
            fact_table['columns'].append({
                "name": f"{dim_name}_key",
                "type": "INTEGER",
                "nullable": False,
                "foreign_key": f"dim_{dim_name}.{dim_name}_key"
            })
        
        # Add facts (measures)
        for fact in facts:
            fact_table['columns'].append({
                "name": fact['name'].lower().replace(' ', '_'),
                "type": fact.get('type', 'NUMERIC'),
                "nullable": fact.get('nullable', True),
                "description": fact.get('description', '')
            })
        
        # Add audit columns
        fact_table['columns'].extend([
            {"name": "created_date", "type": "TIMESTAMP", "nullable": False},
            {"name": "updated_date", "type": "TIMESTAMP", "nullable": False}
        ])
        
        return fact_table
    
    def _design_dimension(self, dimension: Dict) -> Dict:
        """Design a dimension table."""
        dim_name = dimension['name'].lower().replace(' ', '_')
        
        dim_table = {
            "name": f"dim_{dim_name}",
            "type": dimension.get('type', 'Type 2'),  # SCD Type
            "columns": [
                {
                    "name": f"{dim_name}_key",
                    "type": "INTEGER",
                    "nullable": False,
                    "primary_key": True,
                    "description": "Surrogate key"
                }
            ]
        }
        
        # Add natural/business key
        if 'business_key' in dimension:
            dim_table['columns'].append({
                "name": dimension['business_key'],
                "type": "VARCHAR(100)",
                "nullable": False,
                "description": "Natural/Business key"
            })
        
        # Add attributes
        for attr in dimension.get('attributes', []):
            dim_table['columns'].append({
                "name": attr['name'].lower().replace(' ', '_'),
                "type": attr.get('type', 'VARCHAR(255)'),
                "nullable": attr.get('nullable', True),
                "description": attr.get('description', '')
            })
        
        # Add SCD Type 2 columns if needed
        if dimension.get('type') == 'Type 2':
            dim_table['columns'].extend([
                {"name": "valid_from", "type": "DATE", "nullable": False},
                {"name": "valid_to", "type": "DATE", "nullable": True},
                {"name": "is_current", "type": "BOOLEAN", "nullable": False}
            ])
        
        # Add audit columns
        dim_table['columns'].extend([
            {"name": "created_date", "type": "TIMESTAMP", "nullable": False},
            {"name": "updated_date", "type": "TIMESTAMP", "nullable": False}
        ])
        
        return dim_table
    
    def _define_relationships(self, process: str, dimensions: List[Dict]) -> List[Dict]:
        """Define relationships between fact and dimension tables."""
        relationships = []
        fact_table = f"fact_{process.lower().replace(' ', '_')}"
        
        for dim in dimensions:
            dim_name = dim['name'].lower().replace(' ', '_')
            relationships.append({
                "from_table": fact_table,
                "from_column": f"{dim_name}_key",
                "to_table": f"dim_{dim_name}",
                "to_column": f"{dim_name}_key",
                "relationship_type": "many-to-one"
            })
        
        return relationships
    
    def _generate_ddl(self, model: Dict) -> Dict[str, str]:
        """Generate DDL statements for the model."""
        ddl = {}
        
        # Fact table DDL
        fact = model['fact_table']
        fact_ddl = f"CREATE TABLE {fact['name']} (\n"
        for col in fact['columns']:
            fact_ddl += f"    {col['name']} {col['type']}"
            if not col.get('nullable', True):
                fact_ddl += " NOT NULL"
            fact_ddl += ",\n"
        fact_ddl = fact_ddl.rstrip(',\n') + "\n);"
        ddl[fact['name']] = fact_ddl
        
        # Dimension tables DDL
        for dim in model['dimension_tables']:
            dim_ddl = f"CREATE TABLE {dim['name']} (\n"
            for col in dim['columns']:
                dim_ddl += f"    {col['name']} {col['type']}"
                if col.get('primary_key'):
                    dim_ddl += " PRIMARY KEY"
                if not col.get('nullable', True):
                    dim_ddl += " NOT NULL"
                dim_ddl += ",\n"
            dim_ddl = dim_ddl.rstrip(',\n') + "\n);"
            ddl[dim['name']] = dim_ddl
        
        return ddl
    
    def _generate_documentation(self, model: Dict) -> str:
        """Generate documentation for the dimensional model."""
        doc = f"""
# Dimensional Model: {model['business_process']}

## Overview
- **Business Process**: {model['business_process']}
- **Grain**: {model['grain']}
- **Model Type**: {model['model_type']}

## Fact Table
**Table**: `{model['fact_table']['name']}`

### Measures
"""
        
        for col in model['fact_table']['columns']:
            if not col['name'].endswith('_key') and col['name'] not in ['created_date', 'updated_date']:
                doc += f"- `{col['name']}`: {col.get('description', 'Measure')}\n"
        
        doc += "\n## Dimension Tables\n"
        
        for dim in model['dimension_tables']:
            doc += f"\n### {dim['name']}\n"
            doc += f"**Type**: {dim.get('type', 'Type 1')} SCD\n\n"
            doc += "**Attributes**:\n"
            
            for col in dim['columns']:
                if not col.get('primary_key') and col['name'] not in ['created_date', 'updated_date', 'valid_from', 'valid_to', 'is_current']:
                    doc += f"- `{col['name']}`: {col.get('description', 'Attribute')}\n"
        
        return doc