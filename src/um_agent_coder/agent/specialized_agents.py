"""
Specialized Agent Roles for Domain-Specific Tasks
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from abc import ABC, abstractmethod


class DomainExpertise(Enum):
    """Domain expertise categories"""
    BIOTECH = "biotech"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    DATA_SCIENCE = "data_science"
    INFRASTRUCTURE = "infrastructure"


class SpecializedRole(Enum):
    """All specialized agent roles"""
    
    # Biotech & Healthcare
    COMP_BIO = "computational_biologist"
    BIOINFORMATICS = "bioinformatics_engineer"
    GENOMICS = "genomics_analyst"
    PROTEOMICS = "proteomics_specialist"
    DRUG_DISCOVERY = "drug_discovery_scientist"
    CLINICAL_DATA = "clinical_data_analyst"
    BIOSTATISTICIAN = "biostatistician"
    REGULATORY = "regulatory_compliance_specialist"
    MEDICAL_AI = "medical_ai_engineer"
    
    # Financial & Trading
    QUANT_TRADER = "quantitative_trader"
    FINANCIAL_ANALYST = "financial_analyst"
    STOCK_ANALYZER = "stock_analyzer"
    RISK_ANALYST = "risk_analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    ALGO_TRADING = "algorithmic_trading_developer"
    MARKET_RESEARCHER = "market_researcher"
    CRYPTO_ANALYST = "cryptocurrency_analyst"
    COMPLIANCE_OFFICER = "compliance_officer"
    FINANCIAL_ENGINEER = "financial_engineer"
    
    # Technical & Engineering
    FRONTEND_DEV = "frontend_developer"
    BACKEND_DEV = "backend_developer"
    FULLSTACK_DEV = "fullstack_developer"
    DATA_ARCHITECT = "data_architect"
    SYSTEMS_ARCHITECT = "systems_architect"
    DATABASE_ADMIN = "database_administrator"
    DEVOPS_ENGINEER = "devops_engineer"
    CLOUD_ARCHITECT = "cloud_architect"
    SECURITY_ENGINEER = "security_engineer"
    ML_ENGINEER = "machine_learning_engineer"
    DATA_ENGINEER = "data_engineer"
    
    # Data & Analytics
    DATA_SCIENTIST = "data_scientist"
    DATA_ANALYST = "data_analyst"
    BI_ANALYST = "business_intelligence_analyst"
    ETL_DEVELOPER = "etl_developer"
    VISUALIZATION_EXPERT = "data_visualization_expert"


@dataclass
class AgentCapabilities:
    """Define capabilities for specialized agents"""
    role: SpecializedRole
    domain: DomainExpertise
    primary_skills: List[str]
    tools: List[str]
    knowledge_areas: List[str]
    output_formats: List[str]
    complexity_range: Tuple[int, int]  # min, max complexity (1-10)
    preferred_models: List[str]


class SpecializedAgent(ABC):
    """Base class for specialized agents"""
    
    def __init__(self, role: SpecializedRole):
        self.role = role
        self.capabilities = self._init_capabilities()
        self.context = {}
        
    @abstractmethod
    def _init_capabilities(self) -> AgentCapabilities:
        """Initialize agent capabilities"""
        pass
    
    @abstractmethod
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Prepare specialized prompt for the agent's domain"""
        pass
    
    @abstractmethod
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        """Validate output according to domain standards"""
        pass
    
    def get_system_prompt(self) -> str:
        """Get specialized system prompt for this agent"""
        return f"""You are a specialized {self.role.value.replace('_', ' ')} with expertise in {self.capabilities.domain.value}.
Your primary skills include: {', '.join(self.capabilities.primary_skills)}.
You have deep knowledge in: {', '.join(self.capabilities.knowledge_areas)}.
Always provide responses that are technically accurate, domain-specific, and actionable."""


# ============= BIOTECH AGENTS =============

class ComputationalBiologistAgent(SpecializedAgent):
    """Computational Biologist Agent"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.COMP_BIO,
            domain=DomainExpertise.BIOTECH,
            primary_skills=[
                "sequence_analysis", "protein_modeling", "systems_biology",
                "mathematical_modeling", "statistical_analysis"
            ],
            tools=["biopython", "pymol", "rosetta", "blast", "clustal"],
            knowledge_areas=[
                "molecular_biology", "genetics", "biochemistry",
                "biophysics", "evolution", "structural_biology"
            ],
            output_formats=["python_code", "r_code", "jupyter_notebook", "research_report"],
            complexity_range=(5, 10),
            preferred_models=["claude-3-opus", "gpt-4"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a computational biologist, analyze the following task:

Task: {task}

Consider:
1. Biological relevance and accuracy
2. Appropriate computational methods
3. Statistical validation approaches
4. Relevant databases and tools (NCBI, UniProt, PDB, etc.)
5. Reproducibility and documentation

Context: {json.dumps(context, indent=2)}

Provide a comprehensive solution with code, analysis, and biological interpretation."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        # Check for biological accuracy markers
        required_elements = ["biological", "statistical", "p-value", "significance"]
        has_required = any(elem in output.lower() for elem in required_elements)
        
        if not has_required:
            return False, "Output lacks biological or statistical validation"
        return True, None


class BioinformaticsEngineerAgent(SpecializedAgent):
    """Bioinformatics Pipeline Engineer"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.BIOINFORMATICS,
            domain=DomainExpertise.BIOTECH,
            primary_skills=[
                "pipeline_development", "ngs_analysis", "workflow_management",
                "cloud_computing", "containerization"
            ],
            tools=["nextflow", "snakemake", "docker", "kubernetes", "cwl"],
            knowledge_areas=[
                "genomics", "transcriptomics", "variant_calling",
                "alignment_algorithms", "quality_control", "data_formats"
            ],
            output_formats=["pipeline_code", "workflow_diagram", "documentation"],
            complexity_range=(6, 10),
            preferred_models=["claude-3-sonnet", "gpt-4-turbo"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a bioinformatics pipeline engineer, design and implement:

Task: {task}

Requirements:
1. Scalable and reproducible pipeline architecture
2. Proper error handling and checkpointing
3. Resource optimization (CPU, memory, storage)
4. Standard bioinformatics formats (FASTQ, BAM, VCF, etc.)
5. Quality control and validation steps
6. Container/environment specifications

Context: {json.dumps(context, indent=2)}

Provide complete pipeline code with configuration and documentation."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        pipeline_keywords = ["pipeline", "workflow", "step", "input", "output"]
        has_pipeline = any(kw in output.lower() for kw in pipeline_keywords)
        
        if not has_pipeline:
            return False, "Output lacks pipeline structure"
        return True, None


class GenomicsAnalystAgent(SpecializedAgent):
    """Genomics Data Analyst"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.GENOMICS,
            domain=DomainExpertise.BIOTECH,
            primary_skills=[
                "variant_analysis", "gwas", "population_genetics",
                "comparative_genomics", "annotation"
            ],
            tools=["gatk", "plink", "vcftools", "annovar", "igv"],
            knowledge_areas=[
                "human_genetics", "cancer_genomics", "rare_diseases",
                "pharmacogenomics", "evolutionary_genomics"
            ],
            output_formats=["analysis_report", "vcf_files", "visualization"],
            complexity_range=(5, 9),
            preferred_models=["claude-3-opus", "gpt-4"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a genomics analyst, perform the following analysis:

Task: {task}

Focus on:
1. Variant identification and annotation
2. Population frequency analysis
3. Functional impact prediction
4. Clinical relevance assessment
5. Quality metrics and filtering

Context: {json.dumps(context, indent=2)}

Provide detailed genomic analysis with interpretations."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        genomics_terms = ["variant", "snp", "mutation", "allele", "genotype"]
        has_genomics = any(term in output.lower() for term in genomics_terms)
        
        if not has_genomics:
            return False, "Output lacks genomics-specific analysis"
        return True, None


# ============= FINANCIAL AGENTS =============

class QuantitativeTraderAgent(SpecializedAgent):
    """Quantitative Trading Strategist"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.QUANT_TRADER,
            domain=DomainExpertise.FINANCIAL,
            primary_skills=[
                "algorithmic_trading", "statistical_arbitrage", "market_making",
                "risk_modeling", "backtesting"
            ],
            tools=["python", "quantlib", "zipline", "backtrader", "vectorbt"],
            knowledge_areas=[
                "derivatives_pricing", "stochastic_calculus", "time_series",
                "market_microstructure", "portfolio_theory"
            ],
            output_formats=["trading_strategy", "backtest_report", "risk_metrics"],
            complexity_range=(7, 10),
            preferred_models=["gpt-4", "claude-3-opus"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a quantitative trader, develop the following:

Task: {task}

Requirements:
1. Mathematical model and assumptions
2. Trading signals and entry/exit logic
3. Risk management and position sizing
4. Backtesting with performance metrics
5. Sharpe ratio, max drawdown, win rate analysis
6. Market regime considerations

Context: {json.dumps(context, indent=2)}

Provide complete trading strategy with code and analysis."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        trading_terms = ["strategy", "signal", "backtest", "sharpe", "return"]
        has_trading = any(term in output.lower() for term in trading_terms)
        
        if not has_trading:
            return False, "Output lacks trading strategy components"
        return True, None


class FinancialAnalystAgent(SpecializedAgent):
    """Financial Analysis Expert"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.FINANCIAL_ANALYST,
            domain=DomainExpertise.FINANCIAL,
            primary_skills=[
                "financial_modeling", "valuation", "ratio_analysis",
                "forecasting", "budgeting"
            ],
            tools=["excel", "python", "pandas", "numpy", "bloomberg_api"],
            knowledge_areas=[
                "corporate_finance", "accounting", "economics",
                "industry_analysis", "regulatory_compliance"
            ],
            output_formats=["financial_model", "valuation_report", "dashboard"],
            complexity_range=(5, 9),
            preferred_models=["gpt-4-turbo", "claude-3-sonnet"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a financial analyst, provide analysis for:

Task: {task}

Include:
1. Financial statement analysis
2. Key ratios and metrics (ROE, ROA, P/E, etc.)
3. Peer comparison and industry benchmarks
4. DCF or comparable valuation models
5. Risk factors and sensitivities
6. Investment recommendation with rationale

Context: {json.dumps(context, indent=2)}

Deliver comprehensive financial analysis with supporting data."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        financial_terms = ["revenue", "ebitda", "valuation", "ratio", "forecast"]
        has_financial = any(term in output.lower() for term in financial_terms)
        
        if not has_financial:
            return False, "Output lacks financial analysis"
        return True, None


class RiskAnalystAgent(SpecializedAgent):
    """Financial Risk Analyst"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.RISK_ANALYST,
            domain=DomainExpertise.FINANCIAL,
            primary_skills=[
                "var_modeling", "stress_testing", "credit_risk",
                "market_risk", "operational_risk"
            ],
            tools=["python", "r", "matlab", "riskmetrics", "basel_framework"],
            knowledge_areas=[
                "risk_management", "derivatives", "regulation",
                "monte_carlo", "extreme_value_theory"
            ],
            output_formats=["risk_report", "var_calculation", "stress_test_results"],
            complexity_range=(6, 10),
            preferred_models=["gpt-4", "claude-3-opus"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a risk analyst, assess the following:

Task: {task}

Analyze:
1. Value at Risk (VaR) calculations
2. Stress testing scenarios
3. Risk factor sensitivities
4. Correlation and concentration risks
5. Regulatory capital requirements
6. Risk mitigation strategies

Context: {json.dumps(context, indent=2)}

Provide comprehensive risk assessment with quantitative metrics."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        risk_terms = ["risk", "var", "exposure", "volatility", "correlation"]
        has_risk = any(term in output.lower() for term in risk_terms)
        
        if not has_risk:
            return False, "Output lacks risk analysis"
        return True, None


# ============= TECHNICAL AGENTS =============

class FrontendDeveloperAgent(SpecializedAgent):
    """Frontend Development Specialist"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.FRONTEND_DEV,
            domain=DomainExpertise.TECHNICAL,
            primary_skills=[
                "react", "vue", "angular", "typescript", "css",
                "responsive_design", "accessibility", "performance"
            ],
            tools=["webpack", "vite", "jest", "cypress", "storybook"],
            knowledge_areas=[
                "ui_ux", "web_standards", "browser_apis", "state_management",
                "component_architecture", "design_systems"
            ],
            output_formats=["react_component", "vue_component", "html_css", "tests"],
            complexity_range=(3, 8),
            preferred_models=["claude-3-sonnet", "gpt-4-turbo"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a frontend developer, implement:

Task: {task}

Requirements:
1. Modern, responsive UI components
2. Accessibility (WCAG 2.1 AA compliance)
3. Performance optimization
4. Cross-browser compatibility
5. Clean, maintainable code with TypeScript
6. Unit and integration tests

Context: {json.dumps(context, indent=2)}

Provide complete implementation with styling and tests."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        frontend_terms = ["component", "render", "state", "props", "css"]
        has_frontend = any(term in output.lower() for term in frontend_terms)
        
        if not has_frontend:
            return False, "Output lacks frontend implementation"
        return True, None


class BackendDeveloperAgent(SpecializedAgent):
    """Backend Development Specialist"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.BACKEND_DEV,
            domain=DomainExpertise.TECHNICAL,
            primary_skills=[
                "api_design", "database_design", "microservices",
                "authentication", "caching", "message_queues"
            ],
            tools=["nodejs", "python", "golang", "postgresql", "redis", "rabbitmq"],
            knowledge_areas=[
                "rest_api", "graphql", "websockets", "security",
                "scalability", "distributed_systems"
            ],
            output_formats=["api_code", "database_schema", "documentation"],
            complexity_range=(4, 9),
            preferred_models=["claude-3-sonnet", "gpt-4"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a backend developer, design and implement:

Task: {task}

Requirements:
1. RESTful API design with proper status codes
2. Database schema with indexes and constraints
3. Authentication and authorization
4. Input validation and error handling
5. Logging and monitoring
6. Performance optimization and caching
7. API documentation

Context: {json.dumps(context, indent=2)}

Provide complete backend implementation with tests."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        backend_terms = ["api", "endpoint", "database", "query", "authentication"]
        has_backend = any(term in output.lower() for term in backend_terms)
        
        if not has_backend:
            return False, "Output lacks backend implementation"
        return True, None


class DataArchitectAgent(SpecializedAgent):
    """Data Architecture Specialist"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.DATA_ARCHITECT,
            domain=DomainExpertise.INFRASTRUCTURE,
            primary_skills=[
                "data_modeling", "etl_design", "data_warehouse",
                "data_lake", "data_governance", "metadata_management"
            ],
            tools=["snowflake", "databricks", "airflow", "dbt", "kafka"],
            knowledge_areas=[
                "dimensional_modeling", "data_vault", "lambda_architecture",
                "data_quality", "master_data", "compliance"
            ],
            output_formats=["architecture_diagram", "data_model", "etl_pipeline"],
            complexity_range=(6, 10),
            preferred_models=["gpt-4", "claude-3-opus"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a data architect, design:

Task: {task}

Include:
1. Conceptual, logical, and physical data models
2. Data flow and integration patterns
3. Storage and compute architecture
4. Data governance and quality framework
5. Security and compliance measures
6. Scalability and performance considerations
7. Technology stack recommendations

Context: {json.dumps(context, indent=2)}

Provide comprehensive data architecture with implementation guide."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        data_terms = ["schema", "etl", "warehouse", "pipeline", "governance"]
        has_data_arch = any(term in output.lower() for term in data_terms)
        
        if not has_data_arch:
            return False, "Output lacks data architecture design"
        return True, None


class SystemsArchitectAgent(SpecializedAgent):
    """Systems Architecture Specialist"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.SYSTEMS_ARCHITECT,
            domain=DomainExpertise.INFRASTRUCTURE,
            primary_skills=[
                "distributed_systems", "microservices", "cloud_architecture",
                "scalability", "reliability", "security_architecture"
            ],
            tools=["kubernetes", "terraform", "aws", "azure", "gcp"],
            knowledge_areas=[
                "design_patterns", "cap_theorem", "consensus_algorithms",
                "load_balancing", "fault_tolerance", "disaster_recovery"
            ],
            output_formats=["architecture_diagram", "design_document", "deployment_yaml"],
            complexity_range=(7, 10),
            preferred_models=["gpt-4", "claude-3-opus"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a systems architect, design:

Task: {task}

Address:
1. High-level system architecture and components
2. Communication patterns and protocols
3. Scalability and performance requirements
4. Fault tolerance and disaster recovery
5. Security architecture and threat model
6. Deployment and operations strategy
7. Technology choices with rationale

Context: {json.dumps(context, indent=2)}

Provide detailed system architecture with implementation roadmap."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        systems_terms = ["architecture", "scalability", "microservice", "deployment", "reliability"]
        has_systems = any(term in output.lower() for term in systems_terms)
        
        if not has_systems:
            return False, "Output lacks systems architecture design"
        return True, None


class DatabaseAdminAgent(SpecializedAgent):
    """Database Administrator Specialist"""
    
    def _init_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            role=SpecializedRole.DATABASE_ADMIN,
            domain=DomainExpertise.INFRASTRUCTURE,
            primary_skills=[
                "query_optimization", "index_design", "backup_recovery",
                "replication", "partitioning", "performance_tuning"
            ],
            tools=["postgresql", "mysql", "mongodb", "redis", "elasticsearch"],
            knowledge_areas=[
                "acid_properties", "normalization", "sharding",
                "clustering", "monitoring", "security"
            ],
            output_formats=["sql_scripts", "optimization_plan", "backup_strategy"],
            complexity_range=(5, 9),
            preferred_models=["claude-3-sonnet", "gpt-4"]
        )
    
    def prepare_prompt(self, task: str, context: Dict[str, Any]) -> str:
        prompt = f"""As a database administrator, implement:

Task: {task}

Focus on:
1. Schema design and normalization
2. Index strategy and query optimization
3. Backup and recovery procedures
4. Replication and high availability
5. Security and access control
6. Monitoring and alerting
7. Performance tuning recommendations

Context: {json.dumps(context, indent=2)}

Provide complete database solution with scripts and procedures."""
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, Optional[str]]:
        db_terms = ["table", "index", "query", "transaction", "backup"]
        has_db = any(term in output.lower() for term in db_terms)
        
        if not has_db:
            return False, "Output lacks database administration details"
        return True, None


# ============= AGENT FACTORY =============

class SpecializedAgentFactory:
    """Factory for creating specialized agents"""
    
    # Agent class mapping
    AGENT_CLASSES = {
        # Biotech
        SpecializedRole.COMP_BIO: ComputationalBiologistAgent,
        SpecializedRole.BIOINFORMATICS: BioinformaticsEngineerAgent,
        SpecializedRole.GENOMICS: GenomicsAnalystAgent,
        
        # Financial
        SpecializedRole.QUANT_TRADER: QuantitativeTraderAgent,
        SpecializedRole.FINANCIAL_ANALYST: FinancialAnalystAgent,
        SpecializedRole.RISK_ANALYST: RiskAnalystAgent,
        
        # Technical
        SpecializedRole.FRONTEND_DEV: FrontendDeveloperAgent,
        SpecializedRole.BACKEND_DEV: BackendDeveloperAgent,
        SpecializedRole.DATA_ARCHITECT: DataArchitectAgent,
        SpecializedRole.SYSTEMS_ARCHITECT: SystemsArchitectAgent,
        SpecializedRole.DATABASE_ADMIN: DatabaseAdminAgent,
    }
    
    @classmethod
    def create_agent(cls, role: SpecializedRole) -> Optional[SpecializedAgent]:
        """Create a specialized agent by role"""
        agent_class = cls.AGENT_CLASSES.get(role)
        if agent_class:
            return agent_class(role)
        return None
    
    @classmethod
    def get_agents_by_domain(cls, domain: DomainExpertise) -> List[SpecializedAgent]:
        """Get all agents for a specific domain"""
        agents = []
        for role, agent_class in cls.AGENT_CLASSES.items():
            agent = agent_class(role)
            if agent.capabilities.domain == domain:
                agents.append(agent)
        return agents
    
    @classmethod
    def recommend_agent(cls, task: str, context: Dict[str, Any]) -> Optional[SpecializedAgent]:
        """Recommend the best agent for a task"""
        
        # Keywords for agent selection
        biotech_keywords = ["gene", "protein", "dna", "rna", "sequence", "drug", "clinical", "genomics"]
        financial_keywords = ["stock", "trading", "portfolio", "risk", "valuation", "market", "investment"]
        frontend_keywords = ["ui", "react", "vue", "component", "frontend", "css", "responsive"]
        backend_keywords = ["api", "database", "server", "backend", "authentication", "microservice"]
        data_keywords = ["etl", "warehouse", "pipeline", "data model", "analytics", "big data"]
        
        task_lower = task.lower()
        
        # Score each domain
        scores = {}
        
        # Check biotech
        biotech_score = sum(1 for kw in biotech_keywords if kw in task_lower)
        if biotech_score > 0:
            scores[DomainExpertise.BIOTECH] = biotech_score
        
        # Check financial
        financial_score = sum(1 for kw in financial_keywords if kw in task_lower)
        if financial_score > 0:
            scores[DomainExpertise.FINANCIAL] = financial_score
        
        # Check technical
        technical_score = sum(1 for kw in frontend_keywords + backend_keywords if kw in task_lower)
        if technical_score > 0:
            scores[DomainExpertise.TECHNICAL] = technical_score
        
        # Check data/infrastructure
        data_score = sum(1 for kw in data_keywords if kw in task_lower)
        if data_score > 0:
            scores[DomainExpertise.INFRASTRUCTURE] = data_score
        
        if not scores:
            return None
        
        # Get domain with highest score
        best_domain = max(scores.items(), key=lambda x: x[1])[0]
        
        # Get agents for that domain
        domain_agents = cls.get_agents_by_domain(best_domain)
        
        if not domain_agents:
            return None
        
        # Return first matching agent (could be enhanced with more specific selection)
        return domain_agents[0]


# Example usage
def example_usage():
    """Example of using specialized agents"""
    
    # Create a computational biologist
    comp_bio = SpecializedAgentFactory.create_agent(SpecializedRole.COMP_BIO)
    print(f"Created: {comp_bio.role.value}")
    print(f"System Prompt:\n{comp_bio.get_system_prompt()}\n")
    
    # Prepare a biotech task
    biotech_task = "Analyze the protein sequence and predict its 3D structure"
    biotech_prompt = comp_bio.prepare_prompt(biotech_task, {"organism": "human"})
    print(f"Biotech Prompt:\n{biotech_prompt[:200]}...\n")
    
    # Create a quant trader
    quant_trader = SpecializedAgentFactory.create_agent(SpecializedRole.QUANT_TRADER)
    print(f"Created: {quant_trader.role.value}")
    
    # Prepare a trading task
    trading_task = "Develop a pairs trading strategy for tech stocks"
    trading_prompt = quant_trader.prepare_prompt(trading_task, {"market": "NASDAQ"})
    print(f"Trading Prompt:\n{trading_prompt[:200]}...\n")
    
    # Recommend agent for a task
    recommended = SpecializedAgentFactory.recommend_agent(
        "Build a React component for displaying stock charts",
        {}
    )
    if recommended:
        print(f"Recommended agent: {recommended.role.value}")


if __name__ == "__main__":
    example_usage()