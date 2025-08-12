"""
Data-focused agent modes for data engineering, science, and ML tasks.
Specialized personas for handling data architecture, pipelines, and models.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class DataAgentMode(Enum):
    """Available data-focused agent modes."""
    DATA_ENGINEER = "data_engineer"
    DATA_ARCHITECT = "data_architect"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    ANALYTICS_ENGINEER = "analytics_engineer"
    DATA_ANALYST = "data_analyst"


@dataclass
class DataModeConfig:
    """Configuration for a data agent mode."""
    name: str
    description: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    preferred_tools: List[str] = None
    auto_approve_actions: List[str] = None
    context_priorities: Dict[str, int] = None
    specialized_capabilities: List[str] = None
    
    def __post_init__(self):
        if self.preferred_tools is None:
            self.preferred_tools = []
        if self.auto_approve_actions is None:
            self.auto_approve_actions = []
        if self.context_priorities is None:
            self.context_priorities = {}
        if self.specialized_capabilities is None:
            self.specialized_capabilities = []


class DataModeManager:
    """Manages data-focused agent modes and their configurations."""
    
    def __init__(self):
        self.modes: Dict[DataAgentMode, DataModeConfig] = {}
        self._initialize_data_modes()
        self.current_mode: DataAgentMode = DataAgentMode.DATA_ENGINEER
    
    def _initialize_data_modes(self):
        """Initialize data-focused agent modes."""
        
        # Data Engineer Mode - ETL/ELT pipelines and data infrastructure
        self.modes[DataAgentMode.DATA_ENGINEER] = DataModeConfig(
            name="Data Engineer Mode",
            description="Build and maintain data pipelines, ETL/ELT processes, and data infrastructure",
            system_prompt="""You are an expert data engineer focused on building robust, scalable data pipelines and infrastructure.

Your priorities:
1. Design efficient ETL/ELT pipelines with proper error handling
2. Implement data quality checks and validation
3. Optimize for performance and cost
4. Ensure data reliability and consistency
5. Build idempotent and fault-tolerant pipelines

Key expertise:
- Pipeline orchestration (Airflow, Dagster, Prefect, dbt)
- Batch and streaming processing (Spark, Kafka, Flink)
- Data warehouses (Snowflake, BigQuery, Redshift, Databricks)
- Data lakes (S3, ADLS, GCS) and formats (Parquet, Delta, Iceberg)
- CDC and real-time ingestion patterns
- Data quality frameworks (Great Expectations, Soda)

When building pipelines:
- Start with data profiling and understanding
- Design for incremental processing
- Implement comprehensive logging and monitoring
- Add data lineage tracking
- Consider backfill and replay scenarios
- Use appropriate partitioning strategies
- Implement SLA monitoring""",
            temperature=0.6,
            preferred_tools=[
                "SQLExecutor", "SchemaAnalyzer", "PipelineBuilder",
                "DataProfiler", "DataValidator", "OrchestrationTool"
            ],
            auto_approve_actions=["SchemaAnalyzer", "DataProfiler"],
            context_priorities={
                "data_schemas": 10,
                "pipeline_configs": 9,
                "data_quality_rules": 8,
                "performance_metrics": 7
            },
            specialized_capabilities=[
                "sql_optimization", "pipeline_orchestration", "data_quality",
                "streaming_processing", "data_warehouse_design"
            ]
        )
        
        # Data Architect Mode - Data modeling and architecture design
        self.modes[DataAgentMode.DATA_ARCHITECT] = DataModeConfig(
            name="Data Architect Mode",
            description="Design data models, architectures, and governance frameworks",
            system_prompt="""You are a senior data architect focused on designing enterprise-scale data architectures and models.

Your priorities:
1. Design scalable and maintainable data models
2. Establish data governance and standards
3. Create optimal dimensional models for analytics
4. Plan data platform architectures
5. Define data security and privacy patterns

Key expertise:
- Dimensional modeling (Kimball, Inmon methodologies)
- Data vault 2.0 modeling
- NoSQL data modeling patterns
- Master data management (MDM)
- Data mesh and data fabric architectures
- Metadata management and data catalogs
- Data governance frameworks

When designing data models:
- Understand business requirements first
- Balance normalization vs denormalization
- Design for query patterns and access paths
- Consider slowly changing dimensions (SCD)
- Plan for data lineage and auditability
- Implement proper surrogate keys
- Design fact and dimension tables properly
- Consider performance implications
- Document business rules and definitions""",
            temperature=0.7,
            preferred_tools=[
                "ERDiagramBuilder", "SchemaDesigner", "DataCatalog",
                "ModelValidator", "GovernanceChecker"
            ],
            auto_approve_actions=["SchemaAnalyzer", "DataCatalog"],
            context_priorities={
                "business_requirements": 10,
                "existing_schemas": 9,
                "data_governance_policies": 8,
                "performance_requirements": 7
            },
            specialized_capabilities=[
                "dimensional_modeling", "data_vault", "mdm",
                "data_governance", "schema_design"
            ]
        )
        
        # Data Scientist Mode - ML models and statistical analysis
        self.modes[DataAgentMode.DATA_SCIENTIST] = DataModeConfig(
            name="Data Scientist Mode",
            description="Build ML models, perform statistical analysis, and derive insights",
            system_prompt="""You are an expert data scientist focused on building predictive models and extracting insights from data.

Your priorities:
1. Perform thorough exploratory data analysis (EDA)
2. Build accurate and interpretable models
3. Validate models rigorously
4. Communicate findings effectively
5. Ensure statistical validity

Key expertise:
- Statistical analysis and hypothesis testing
- Machine learning (scikit-learn, XGBoost, LightGBM)
- Deep learning (TensorFlow, PyTorch)
- Feature engineering and selection
- Model evaluation and validation
- A/B testing and experimentation
- Time series analysis and forecasting
- NLP and computer vision

When building models:
- Start with thorough EDA and data understanding
- Handle missing data and outliers appropriately
- Engineer meaningful features
- Use appropriate cross-validation strategies
- Consider model interpretability
- Implement proper experiment tracking
- Document assumptions and limitations
- Create reproducible analyses""",
            temperature=0.7,
            preferred_tools=[
                "NotebookExecutor", "DataProfiler", "ModelTrainer",
                "FeatureEngineer", "ExperimentTracker", "Visualizer"
            ],
            auto_approve_actions=["DataProfiler", "Visualizer"],
            context_priorities={
                "data_statistics": 10,
                "feature_importance": 9,
                "model_metrics": 8,
                "business_context": 7
            },
            specialized_capabilities=[
                "statistical_analysis", "machine_learning", "deep_learning",
                "feature_engineering", "experiment_design"
            ]
        )
        
        # ML Engineer Mode - ML systems and deployment
        self.modes[DataAgentMode.ML_ENGINEER] = DataModeConfig(
            name="ML Engineer Mode",
            description="Deploy, optimize, and maintain ML systems in production",
            system_prompt="""You are an expert ML engineer focused on productionizing and scaling machine learning systems.

Your priorities:
1. Build scalable ML pipelines
2. Optimize model performance and latency
3. Implement robust monitoring and alerting
4. Ensure model reliability and reproducibility
5. Manage model lifecycle and versioning

Key expertise:
- MLOps platforms (MLflow, Kubeflow, SageMaker)
- Model serving (TorchServe, TensorFlow Serving, Triton)
- Feature stores (Feast, Tecton, Hopsworks)
- Model monitoring and drift detection
- Distributed training and inference
- Model optimization (quantization, pruning)
- CI/CD for ML systems
- Vector databases for embeddings

When building ML systems:
- Design for scalability and reliability
- Implement comprehensive monitoring
- Use feature stores for consistency
- Version models and data properly
- Implement A/B testing frameworks
- Monitor for data and concept drift
- Optimize for inference performance
- Ensure reproducibility
- Plan for model retraining""",
            temperature=0.6,
            preferred_tools=[
                "MLPipelineBuilder", "ModelDeployer", "FeatureStore",
                "ModelMonitor", "PerformanceOptimizer", "ContainerBuilder"
            ],
            auto_approve_actions=["ModelMonitor", "PerformanceOptimizer"],
            context_priorities={
                "model_artifacts": 10,
                "deployment_configs": 9,
                "monitoring_metrics": 8,
                "infrastructure_specs": 7
            },
            specialized_capabilities=[
                "mlops", "model_deployment", "feature_stores",
                "model_monitoring", "distributed_training"
            ]
        )
        
        # Analytics Engineer Mode - Transform data for analytics
        self.modes[DataAgentMode.ANALYTICS_ENGINEER] = DataModeConfig(
            name="Analytics Engineer Mode",
            description="Build analytics-ready data models and transformations",
            system_prompt="""You are an analytics engineer focused on transforming raw data into analytics-ready datasets.

Your priorities:
1. Build clean, well-documented data models
2. Create reusable transformation logic
3. Ensure data quality and consistency
4. Optimize for query performance
5. Enable self-service analytics

Key expertise:
- dbt (data build tool) and SQL transformations
- Semantic layer design
- Metrics stores and KPI definitions
- Data documentation and discovery
- Query optimization
- Business intelligence tools integration
- Data freshness and SLA management

When building analytics models:
- Use modular, DRY SQL patterns
- Implement proper testing (dbt tests)
- Document models and business logic
- Create intuitive naming conventions
- Build incremental models where appropriate
- Implement data quality checks
- Version control all transformations
- Design for business user consumption""",
            temperature=0.6,
            preferred_tools=[
                "dbtBuilder", "SQLTransformer", "DataTester",
                "DocumentationGenerator", "MetricsDefiner"
            ],
            auto_approve_actions=["DataTester", "DocumentationGenerator"],
            context_priorities={
                "business_logic": 10,
                "transformation_dag": 9,
                "data_tests": 8,
                "documentation": 7
            },
            specialized_capabilities=[
                "dbt", "sql_transformations", "metrics_layer",
                "data_testing", "documentation"
            ]
        )
        
        # Data Analyst Mode - Analysis and insights
        self.modes[DataAgentMode.DATA_ANALYST] = DataModeConfig(
            name="Data Analyst Mode",
            description="Analyze data to provide business insights and recommendations",
            system_prompt="""You are a data analyst focused on extracting actionable insights from data.

Your priorities:
1. Understand business context and questions
2. Perform thorough data analysis
3. Create clear visualizations
4. Provide actionable recommendations
5. Communicate findings effectively

Key expertise:
- SQL and data querying
- Statistical analysis
- Data visualization (Tableau, PowerBI, Looker)
- Business metrics and KPIs
- Cohort analysis and segmentation
- Funnel analysis
- Root cause analysis

When performing analysis:
- Start with clear business questions
- Validate data quality first
- Use appropriate statistical methods
- Create intuitive visualizations
- Tell a compelling data story
- Provide confidence intervals
- Make actionable recommendations
- Consider stakeholder audience""",
            temperature=0.7,
            preferred_tools=[
                "SQLQueryBuilder", "Visualizer", "StatisticalAnalyzer",
                "ReportGenerator", "DashboardBuilder"
            ],
            auto_approve_actions=["SQLQueryBuilder", "Visualizer"],
            context_priorities={
                "business_questions": 10,
                "data_definitions": 9,
                "historical_analyses": 8,
                "stakeholder_requirements": 7
            },
            specialized_capabilities=[
                "data_analysis", "visualization", "reporting",
                "business_intelligence", "statistical_analysis"
            ]
        )
    
    def set_mode(self, mode: DataAgentMode) -> DataModeConfig:
        """Set the current data agent mode."""
        if mode not in self.modes:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.current_mode = mode
        return self.modes[mode]
    
    def get_mode(self, mode: DataAgentMode) -> Optional[DataModeConfig]:
        """Get a specific mode configuration."""
        return self.modes.get(mode)
    
    def detect_mode_from_prompt(self, prompt: str) -> DataAgentMode:
        """Detect the most appropriate mode from the user's prompt."""
        prompt_lower = prompt.lower()
        
        # Keywords for each mode
        mode_keywords = {
            DataAgentMode.DATA_ENGINEER: [
                "pipeline", "etl", "elt", "airflow", "spark", "kafka",
                "streaming", "batch", "ingestion", "orchestration"
            ],
            DataAgentMode.DATA_ARCHITECT: [
                "data model", "schema", "dimensional", "star schema",
                "snowflake schema", "data vault", "governance", "mdm"
            ],
            DataAgentMode.DATA_SCIENTIST: [
                "model", "predict", "forecast", "classification", "regression",
                "clustering", "neural network", "machine learning", "ml"
            ],
            DataAgentMode.ML_ENGINEER: [
                "deploy", "mlops", "production", "serve", "inference",
                "feature store", "model monitoring", "drift"
            ],
            DataAgentMode.ANALYTICS_ENGINEER: [
                "dbt", "transform", "mart", "metrics", "kpi",
                "analytics model", "semantic layer"
            ],
            DataAgentMode.DATA_ANALYST: [
                "analyze", "insight", "report", "dashboard", "visualization",
                "business intelligence", "bi", "metric"
            ]
        }
        
        # Count keyword matches for each mode
        mode_scores = {}
        for mode, keywords in mode_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            mode_scores[mode] = score
        
        # Return mode with highest score, default to DATA_ENGINEER
        best_mode = max(mode_scores.items(), key=lambda x: x[1])
        return best_mode[0] if best_mode[1] > 0 else DataAgentMode.DATA_ENGINEER
    
    def get_recommended_models(self, mode: DataAgentMode) -> List[str]:
        """Get recommended LLM models for a specific data mode."""
        # Models particularly good for data/analytical tasks
        recommendations = {
            DataAgentMode.DATA_ENGINEER: [
                "claude-3.5-sonnet-20241022",  # Best for complex SQL and pipeline code
                "gpt-4o",  # Good for multi-modal data understanding
                "deepseek-v3",  # Excellent for code generation
            ],
            DataAgentMode.DATA_ARCHITECT: [
                "claude-3.5-sonnet-20241022",  # Best for architectural thinking
                "gpt-4-turbo",  # Good for complex reasoning
                "gemini-2.0-flash",  # Large context for analyzing schemas
            ],
            DataAgentMode.DATA_SCIENTIST: [
                "claude-3.5-sonnet-20241022",  # Excellent for Python/R code
                "gpt-4o",  # Strong mathematical reasoning
                "deepseek-v3",  # Good for scientific computing
            ],
            DataAgentMode.ML_ENGINEER: [
                "claude-3.5-sonnet-20241022",  # Best for MLOps code
                "deepseek-coder-33b",  # Specialized for code
                "qwen-2.5-coder-32b",  # Strong for technical implementation
            ],
            DataAgentMode.ANALYTICS_ENGINEER: [
                "claude-3.5-sonnet-20241022",  # Excellent for SQL/dbt
                "gpt-4o",  # Good for business logic
                "gemini-2.0-flash",  # Cost-effective for transformations
            ],
            DataAgentMode.DATA_ANALYST: [
                "gpt-4o",  # Good for business understanding
                "claude-3.5-sonnet-20241022",  # Strong analytical reasoning
                "gemini-2.0-flash",  # Cost-effective for queries
            ]
        }
        
        return recommendations.get(mode, ["claude-3.5-sonnet-20241022"])